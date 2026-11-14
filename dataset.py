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


# ---------------------------------------------------------------------------
# Skeleton images (CNN / ViT baselines): H+P landmarks of all T frames are drawn
# into ONE 224x224 image, colour-coded by time (viridis), so a 2D image model
# sees the same information as the landmark transformer (x, y only).
# ---------------------------------------------------------------------------
def _conns():
    import mediapipe as mp

    pose = [(a, b) for a, b in mp.solutions.pose.POSE_CONNECTIONS if a < 25 and b < 25]
    hand = [(a, b) for a, b in mp.solutions.hands.HAND_CONNECTIONS]
    return pose, hand


_POSE_C, _HAND_C = _conns()
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def render_skeleton(hands, pose, S=224):
    """hands: (T,42,2), pose: (T,25,2), shoulder-normalised. Returns (3,S,S) float32."""
    import cv2

    img = np.zeros((S, S, 3), np.uint8)
    T = len(pose)
    cols = cv2.applyColorMap(np.linspace(40, 255, T).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS)
    cols = cols.reshape(-1, 3).tolist()

    def px(p):  # x in [-3,3], y in [-2.5,3.5] (shoulder widths) -> pixels
        xy = np.stack([(p[:, 0] + 3) / 6, (p[:, 1] + 2.5) / 6], 1) * S
        return np.clip(xy, 0, S - 1).astype(np.int32).tolist()

    for t in range(T):
        for arr, lo, hi, conns in (
            (pose[t], 0, 25, _POSE_C), (hands[t], 0, 21, _HAND_C), (hands[t], 21, 42, _HAND_C)
        ):
            p = arr[lo:hi]
            if not p.any():  # missing hand / dropped frame
                continue
            xy = px(p)
            for a, b in conns:
                cv2.line(img, tuple(xy[a]), tuple(xy[b]), cols[t], 1)
    x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return (x - _MEAN) / _STD


class ImageDataset(SignDataset):
    def __init__(self, df, mods="HP", T=32, train=False, seed=0, **kw):
        super().__init__(df, "HP", T, train, seed, **kw)

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        h = x["hands"].numpy().reshape(self.T, -1, 2)
        p = x["pose"].numpy().reshape(self.T, -1, 2)
        return {"img": torch.from_numpy(render_skeleton(h, p))}, y


# ---------------------------------------------------------------------------
# RGB clips for I3D. 32 uniformly sampled frames (inside the gloss's frame_start/end),
# short-side resized to 176 and centre-cropped, cached once as uint8 .npy.
# ---------------------------------------------------------------------------
FR = LM.parent / "frames"
_KM = np.array([0.45, 0.45, 0.45], np.float32)[:, None, None, None]
_KS = np.array([0.225, 0.225, 0.225], np.float32)[:, None, None, None]


def cache_frames(row, T=32, S=176):
    import cv2

    out = FR / f"{row.video_id}.npy"
    if out.exists():
        return out
    FR.mkdir(parents=True, exist_ok=True)
    start = max(int(row.frame_start) - 1, 0)
    end = int(row.frame_end) - 1 if int(row.frame_end) > 0 else None
    cap = cv2.VideoCapture(str(row.path))
    fr, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (end is not None and i > end):
            break
        if i >= start:
            fr.append(f)
        i += 1
    cap.release()
    if not fr:
        raise RuntimeError(f"no frames for {row.video_id}")
    sel = np.linspace(0, len(fr) - 1, T).round().astype(int)
    clip = []
    for j in sel:
        f = cv2.cvtColor(fr[j], cv2.COLOR_BGR2RGB)
        h, w = f.shape[:2]
        s = S / min(h, w)
        f = cv2.resize(f, (max(S, round(w * s)), max(S, round(h * s))))
        h, w = f.shape[:2]
        y0, x0 = (h - S) // 2, (w - S) // 2
        clip.append(f[y0 : y0 + S, x0 : x0 + S])
    tmp = FR / f"{row.video_id}.tmp.npy"
    np.save(tmp, np.stack(clip))
    tmp.replace(out)
    return out


class VideoDataset(Dataset):
    def __init__(self, df, mods=None, T=32, train=False, seed=0, crop=160, **kw):
        assert T == 32, "I3D frame cache is built for T=32"
        from tqdm import tqdm

        self.train, self.crop, self.T = train, crop, T
        self.rng = np.random.default_rng(seed)
        self.vids = [str(v) for v in df.video_id]
        self.y = df.label.values
        self.dims = None
        self.paths = [cache_frames(r, T) for r in tqdm(df.itertuples(), total=len(df), desc="frames")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        v = np.load(self.paths[i])  # (T,S,S,3) uint8
        S, c = v.shape[1], self.crop
        if self.train:
            y0, x0 = self.rng.integers(0, S - c + 1, 2)
        else:
            y0 = x0 = (S - c) // 2
        v = v[:, y0 : y0 + c, x0 : x0 + c].transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        return {"vid": torch.from_numpy((v - _KM) / _KS)}, int(self.y[i])
