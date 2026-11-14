import sys
import cv2, numpy as np, pandas as pd
from config import LM, SPLIT_CSV, RUNS

vid = sys.argv[1]
df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
row = df[df.video_id == vid].iloc[0]
z = np.load(LM / f"{vid}.npz")
n = len(z["pose"])
for k in ("lh", "rh", "pose", "face"):
    print(k, "NaN frames:", f"{np.isnan(z[k][:, 0, 0].astype(np.float32)).mean():.0%}")
cap = cv2.VideoCapture(row.path)
cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(row.frame_start) - 1, 0) + n // 2)
ok, f = cap.read()
h, w = f.shape[:2]
t = n // 2
for k, col in (("lh", (0, 255, 0)), ("rh", (255, 0, 0)), ("pose", (0, 255, 255))):
    for x, y, _ in z[k][t].astype(np.float32):
        if not np.isnan(x):
            cv2.circle(f, (int(x * w), int(y * h)), 3, col, -1)
out = RUNS / f"check_{vid}.jpg"
cv2.imwrite(str(out), f)
print("wrote", out, "| gloss:", row.gloss, "| signer:", row.signer_id)