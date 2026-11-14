import torch
import torch.nn as nn


class Fuse(nn.Module):
    def __init__(self, stream_dims, d):
        super().__init__()
        self.names = list(stream_dims)
        self.proj = nn.ModuleDict(
            {
                k: nn.Sequential(nn.Linear(v, d), nn.LayerNorm(d))
                for k, v in stream_dims.items()
            }
        )
        self.fuse = nn.Linear(d * len(self.names), d)

    def forward(self, x):
        return self.fuse(torch.cat([self.proj[k](x[k]) for k in self.names], -1))


class SignTransformer(nn.Module):
    def __init__(
        self, stream_dims, n_classes, T=32, d=256, layers=4, heads=8, drop=0.1
    ):
        super().__init__()
        self.fuse = Fuse(stream_dims, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, heads, d * 4, drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.fuse(x)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos
        return self.head(self.norm(self.enc(h)[:, 0]))


class SignLSTM(nn.Module):
    def __init__(self, stream_dims, n_classes, T=32, d=256, layers=2, drop=0.3, **_):
        super().__init__()
        self.fuse = Fuse(stream_dims, d)
        self.rnn = nn.LSTM(
            d, 256, layers, batch_first=True, bidirectional=True, dropout=drop
        )
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(512, n_classes)

    def forward(self, x):
        o, _ = self.rnn(self.fuse(x))
        return self.head(self.drop(o.mean(1)))


class SkeletonImageNet(nn.Module):
    """ImageNet-pretrained CNN (ResNet-18) or ViT-B/16 on rendered skeleton images."""

    def __init__(self, kind, n_classes):
        super().__init__()
        import torchvision.models as tvm

        if kind == "cnn":
            self.net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))
        else:
            self.net = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
            self.net.heads.head = nn.Linear(768, n_classes)

    def forward(self, x):
        return self.net(x["img"])


class I3D(nn.Module):
    """Kinetics-400 pretrained I3D (pytorchvideo i3d_r50), new linear head, any clip length."""

    def __init__(self, n_classes):
        super().__init__()
        net = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
        self.body = nn.Sequential(*net.blocks[:-1])
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(2048, n_classes)

    def forward(self, x):
        return self.head(self.drop(self.body(x["vid"]).mean((2, 3, 4))))


class GC(nn.Module):
    """Graph conv with a fully learnable adjacency (as in the WLASL Pose-TGCN)."""

    def __init__(self, fin, fout, n, drop=0.3):
        super().__init__()
        self.w = nn.Linear(fin, fout, bias=False)
        stdv = 1.0 / fout**0.5
        self.A = nn.Parameter(torch.empty(n, n).uniform_(-stdv, stdv))
        self.bn = nn.BatchNorm1d(n * fout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # (B,N,F)
        h = torch.matmul(self.A, self.w(x))
        h = self.bn(h.flatten(1)).view_as(h)
        return self.drop(torch.tanh(h))


class PoseTGCN(nn.Module):
    """Our re-implementation of Pose-TGCN (Li et al., WACV'20): every keypoint is a graph node whose
    feature is its (x, y) trajectory over all T frames; residual GC blocks; linear classifier.
    Nodes: 25 upper-body pose + 42 hand keypoints (MediaPipe)."""

    def __init__(self, stream_dims, n_classes, T=32, hid=128, blocks=4, **_):
        super().__init__()
        self.n = (stream_dims["pose"] + stream_dims["hands"]) // 2
        self.gc_in = GC(2 * T, hid, self.n)
        self.blocks = nn.ModuleList(
            nn.Sequential(GC(hid, hid, self.n), GC(hid, hid, self.n)) for _ in range(blocks)
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.n * hid, n_classes))

    def forward(self, x):
        B, T, _ = x["pose"].shape
        p = torch.cat([x["pose"], x["hands"]], -1).view(B, T, self.n, 2)
        h = self.gc_in(p.permute(0, 2, 1, 3).reshape(B, self.n, T * 2))
        for b in self.blocks:
            h = h + b(h)
        return self.head(h.flatten(1))


def build_model(name, stream_dims, n_classes, T):
    if name in ("cnn", "vit"):
        return SkeletonImageNet(name, n_classes)
    if name == "i3d":
        return I3D(n_classes)
    cls = {"transformer": SignTransformer, "lstm": SignLSTM, "tgcn": PoseTGCN}[name]
    return cls(stream_dims, n_classes, T=T)
