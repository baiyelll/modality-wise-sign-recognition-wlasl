"""Collects runs/*/metrics.json into the tables used in the report (mean +- sample std over seeds).

  python aggregate.py            # prints all tables, writes runs/summary.csv
"""
import json
import numpy as np, pandas as pd
from config import RUNS

rows = []
for d in sorted(RUNS.glob("*/metrics.json")):
    m, c = json.load(open(d)), json.load(open(d.parent / "config.json"))
    rows.append(dict(run=d.parent.name, model=c["model"], mods=c["mods"], T=c["T"], seed=c["seed"],
                     split=c.get("split", "si"), **{k: 100 * m[k] for k in
                     ("inst_top1", "cls_top1", "inst_top5", "cls_top5", "val_top1")},
                     params=m["n_params"], train_s=m["train_time_s"], ms=m.get("infer_ms_per_clip", np.nan),
                     best_epoch=m["best_epoch"]))
R = pd.DataFrame(rows)
if R.empty:
    raise SystemExit("no finished runs")
R.to_csv(RUNS / "summary_per_run.csv", index=False)
K = ["inst_top1", "cls_top1", "inst_top5", "cls_top5"]


def table(df, by):
    g = df.groupby(by, sort=False)
    out = g[K].agg(["mean", "std"])
    res = pd.DataFrame(index=out.index)
    for k in K:
        res[k] = [f"{m:.1f} ± {0 if np.isnan(s) else s:.1f}" for m, s in zip(out[k]["mean"], out[k]["std"])]
    res["n_seeds"] = g.size()
    res["val_top1"] = g.val_top1.mean().round(1)
    return res


def show(title, df, by):
    if len(df):
        print(f"\n=== {title}\n{table(df, by).to_string()}")


si = R[R.split == "si"]
tf = si[(si.model == "transformer") & (si["T"] == 32)].copy()
tf["o"] = tf.mods.map({m: i for i, m in enumerate(["H", "HP", "HPF", "HPFL", "HPFLD"])}).fillna(9)
show("Table 2: modality ablation (transformer, T=32, signer-independent)", tf.sort_values("o"), "mods")
if "o" in tf and (tf.o == 9).any():
    show("extra mod sets (leave-one-out etc.)", tf[tf.o == 9], "mods")

# best mods by validation
best = None
if len(tf[tf.o < 9]):
    best = tf[tf.o < 9].groupby("mods").val_top1.mean().idxmax()
    print(f"\nbest mod set by mean validation top-1: {best}")

if best:
    fr = si[(si.model == "transformer") & (si.mods == best)]
    fr = fr.assign(seed0=fr.seed == 0)
    f0 = fr[fr.seed == 0].sort_values("T")
    f0 = f0.assign(ms=f0.ms)
    print("\n=== Frame count (seed 0, mods=%s)" % best)
    print(f0[["T", "inst_top1", "cls_top1", "inst_top5", "ms", "train_s"]].round(1).to_string(index=False))

    print("\n=== Table 3: model comparison (signer-independent test)")
    t3 = []
    for label, sel in [
        ("Transformer (ours)", (si.model == "transformer") & (si.mods == best) & (si["T"] == 32)),
        ("Transformer seed 0 only", (si.model == "transformer") & (si.mods == best) & (si["T"] == 32) & (si.seed == 0)),
        ("BiLSTM", si.model == "lstm"), ("CNN (ResNet-18)", si.model == "cnn"), ("ViT-B/16", si.model == "vit"),
        ("I3D (our split)", si.model == "i3d"), ("Pose-TGCN (our split)", si.model == "tgcn"),
    ]:
        d = si[sel]
        if len(d):
            t = table(d.assign(g=label), "g")
            t3.append(t)
    if t3:
        print(pd.concat(t3).to_string())

    sd = R[(R.split == "official") & (R.model == "transformer") & (R.mods == best)]
    if len(sd):
        a, b = sd.inst_top1.mean(), si[(si.model == "transformer") & (si.mods == best) & (si["T"] == 32)].inst_top1.mean()
        print(f"\n=== Signer-dependent (official) vs signer-independent, mods={best}")
        print(table(sd.assign(g="official (signer-dependent)"), "g").to_string())
        print(f"gap (official - ours) top-1: {a - b:+.1f} points")

    # per-signer accuracy of the best model
    P = []
    for r in tf[tf.mods == best].run:
        p = pd.read_csv(RUNS / r / "test_preds.csv")
        p["ok"] = p.label == p.pred
        P.append(p.groupby("signer_id").ok.agg(["mean", "size"]).assign(run=r))
    if P:
        S = pd.concat(P).reset_index().groupby("signer_id").agg(acc=("mean", "mean"), videos=("size", "first"))
        S["acc"] *= 100
        print(f"\n=== Per-test-signer top-1 (mean over seeds), mods={best}")
        print(S.sort_values("acc").round(1).to_string())
        print(f"range {S.acc.min():.1f}% -> {S.acc.max():.1f}%")

R.to_csv(RUNS / "summary_per_run.csv", index=False)
print("\nwrote", RUNS / "summary_per_run.csv")
