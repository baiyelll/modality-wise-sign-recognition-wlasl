"""Numbers for the Data section: clips flagged for >40% frames with a missing hand, extraction failures."""
import numpy as np, pandas as pd
from config import LM, SPLIT_CSV

df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
rows = []
for v in df.video_id:
    f = LM / f"{v}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    miss = lambda k: np.isnan(z[k][:, 0, 0].astype(np.float32))
    lh, rh = miss("lh"), miss("rh")
    rows.append(dict(video_id=v, frames=len(lh), lh_missing=lh.mean(), rh_missing=rh.mean(),
                     both_missing=(lh & rh).mean(), any_hand_missing=(lh | rh).mean(),
                     pose_missing=miss("pose").mean(), face_missing=miss("face").mean()))
S = pd.DataFrame(rows).merge(df[["video_id", "split"]])
print(f"clips with landmarks: {len(S)}/{len(df)}")
# paper: "flag clips in which a hand is missing for more than 40% of frames"
for name, col in (("either hand missing >40% frames", "any_hand_missing"),
                  ("BOTH hands missing >40% frames", "both_missing")):
    fl = S[S[col] > 0.4]
    print(f"{name}: {len(fl)} ({100*len(fl)/len(S):.1f}%) | by split:", fl.split.value_counts().to_dict())
print(S[["frames", "lh_missing", "rh_missing", "pose_missing", "face_missing"]].describe().round(3).to_string())
S.to_csv(LM.parent / "landmark_quality.csv", index=False)
