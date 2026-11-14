import json, sys
import numpy as np
import pandas as pd
from config import RAW, META_CSV, SPLIT_CSV


def find(name):
    hits = sorted(RAW.rglob(name))
    if not hits:
        sys.exit(f"{name} not found under {RAW}")
    return hits[0]


def nid(x):
    return str(x).zfill(5)


meta = json.load(open(find("WLASL_v0.3.json")))
nslt = {nid(k): v for k, v in json.load(open(find("nslt_100.json"))).items()}
videos = {nid(p.stem): p for p in (RAW / "videos").glob("*.mp4")}
print("mp4 files on disk:", len(videos))

info = {}
for e in meta:
    for ins in e["instances"]:
        info[nid(ins["video_id"])] = dict(
            gloss=e["gloss"],
            signer_id=ins["signer_id"],
            official_split=ins["split"],
            frame_start=ins["frame_start"],
            frame_end=ins["frame_end"],
            fps=ins.get("fps"),
        )

rows, missing, no_meta = [], 0, 0
for vid, v in nslt.items():
    if vid not in info:
        no_meta += 1
        continue
    if vid not in videos:
        missing += 1
        continue
    rows.append(
        dict(video_id=vid, path=str(videos[vid]), label=v["action"][0], **info[vid])
    )
df = pd.DataFrame(rows)
print(
    f"WLASL100 listed: {len(nslt)} | found: {len(df)} | missing files: {missing} | no metadata: {no_meta}"
)

bad = (df.signer_id < 0).sum()
if bad:
    print(f"WARNING: dropping {bad} videos with unknown signer_id")
    df = df[df.signer_id >= 0].reset_index(drop=True)

assert df.groupby("gloss").label.nunique().max() == 1, "gloss->label not 1:1"
print("glosses:", df.gloss.nunique(), "| signers:", df.signer_id.nunique())
df.to_csv(META_CSV, index=False)


def make_split(df, seed=0, te=0.15, va=0.10, tries=50000):
    rng = np.random.default_rng(seed)
    signers = np.array(sorted(df.signer_id.unique()))
    glosses = sorted(df.gloss.unique())
    M = (
        pd.crosstab(df.gloss, df.signer_id)
        .reindex(index=glosses, columns=signers, fill_value=0)
        .values
        > 0
    )
    per_signer = df.groupby("signer_id").size().reindex(signers).values
    n_te, n_va = max(1, round(len(signers) * te)), max(1, round(len(signers) * va))
    best, best_v = None, -1
    for _ in range(tries):
        p = rng.permutation(len(signers))
        te_i, va_i, tr_i = p[:n_te], p[n_te : n_te + n_va], p[n_te + n_va :]
        if not (M[:, tr_i].any(1).all() and M[:, te_i].any(1).all()):
            continue
        ft, fv = per_signer[te_i].sum() / len(df), per_signer[va_i].sum() / len(df)
        if not (0.12 <= ft <= 0.22 and 0.08 <= fv <= 0.15):
            continue
        v = M[:, va_i].any(1).sum()
        if v > best_v:
            best_v, best = v, (signers[tr_i], signers[va_i], signers[te_i])
        if v == len(glosses):
            break
    if best is None:
        sys.exit(
            "No valid split found; relax the video-fraction or coverage constraints."
        )
    print(f"glosses covered in val: {best_v}/{len(glosses)}")
    tr, va_, te_ = best
    m = {
        **{s: "train" for s in tr},
        **{s: "val" for s in va_},
        **{s: "test" for s in te_},
    }
    return df.signer_id.map(m)


df["split"] = make_split(df, seed=0)
sets = [set(df[df.split == s].signer_id) for s in ("train", "val", "test")]
assert not (
    sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
), "signer leakage!"
print(
    df.groupby("split").agg(
        videos=("video_id", "count"),
        signers=("signer_id", "nunique"),
        glosses=("gloss", "nunique"),
    )
)
df.to_csv(SPLIT_CSV, index=False)
print("saved", SPLIT_CSV)
