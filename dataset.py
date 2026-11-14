import numpy as np
import torch
from torch.utils.data import Dataset
from config import LM, face_indices

BE, LIPS = face_indices()
STREAMS = {"H": "hands", "P": "pose", "F": "brows_eyes", "L": "lips"}
# D = add the z (depth) coordinate of every included landmark; without D only (x, y) are used.


def interp_nan(a, fill=0.0):
    a = a.copy()
    t = np.arange(len(a))
    for j in range(a.shape[1]):
        m = np.isnan(a[:, j])
        if m.all():
            a[:, j] = fill
        elif m.any():
            a[m, j] = np.interp(t[m], t[~m], a[~m, j])
    return a


def load_clip(vid, mods):
    z = np.load(LM / f"{vid}.npz")
    lh, rh, pose, face = [z[k].astype(np.float32) for k in ("lh", "rh", "pose", "face")]
    c = interp_nan((pose[:, 11] + pose[:, 12]) / 2)  # shoulder midpoint
    sw = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1)
    s = np.nanmedian(sw) if (~np.isnan(sw)).any() else np.nan
    if np.isnan(s) or s < 1e-3:
        s = 1.0

    def norm(x):
        return (x - c[:, None, :]) / s

    lh, rh, pose, face = norm(lh), norm(rh), norm(pose), norm(face)
    raw = {
        "hands": np.concatenate([lh, rh], 1),
        "pose": pose,
        "brows_eyes": face[:, BE],
        "lips": face[:, LIPS],
    }
    out = {}
    for m in "HPFL":
        if m in mods:
            k = STREAMS[m]
            a = raw[k]
            T, N, _ = a.shape
            out[k] = interp_nan(a.reshape(T, -1)).reshape(T, N, 3)
    return out


class SignDataset(Dataset):
    def __init__(self, df, mods, T=32, train=False, seed=0, shuffle_labels=False):
        self.mods, self.T, self.train = mods, T, train
        self.c = 3 if "D" in mods else 2
        self.rng = np.random.default_rng(seed)
        self.items = [load_clip(str(v), mods) for v in df.video_id]
        self.vids = [str(v) for v in df.video_id]
        self.y = df.label.values.copy()
        if shuffle_labels:
            np.random.default_rng(123).shuffle(self.y)
        first = self.items[0]
        self.dims = {k: v.shape[1] * self.c for k, v in first.items()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip = self.items[i]
        L = len(next(iter(clip.values())))
        if self.train:
            span = max(2, int(L * self.rng.uniform(0.8, 1.0)))
            st = self.rng.integers(0, max(L - span, 0) + 1)
            idx = np.linspace(st, min(st + span - 1, L - 1), self.T).round().astype(int)
            ang = np.deg2rad(self.rng.uniform(-15, 15))
            sc = self.rng.uniform(0.9, 1.1)
            R = np.array(
                [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32
            )
        else:
            idx = np.linspace(0, L - 1, self.T).round().astype(int)
        out = {}
        for k, v in clip.items():
            v = v[idx].copy()
            if self.train:
                v[..., :2] = v[..., :2] @ R.T
                v *= sc
                v[self.rng.random(self.T) < 0.1] = 0  # landmark (frame) dropout
            out[k] = torch.from_numpy(v[..., : self.c].reshape(self.T, -1)).float()
        return out, int(self.y[i])
