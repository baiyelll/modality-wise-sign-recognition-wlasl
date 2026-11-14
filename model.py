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


def build_model(name, stream_dims, n_classes, T):
    cls = {"transformer": SignTransformer, "lstm": SignLSTM}[name]
    return cls(stream_dims, n_classes, T=T)
