"""Report figures + extra tables from ../runs. Writes PNGs and figs_summary.txt into ../runs/figs/.

Fig 2: training curves (val top-1 and train acc vs epoch) per modality set, mean over seeds.
Fig 3: most-confused sign pairs for the best modality set (summed over seeds), plus a confusion heat-map.
Fig 4: per-test-signer top-1 for the best modality set, and the spread (min/max over signers) for every modality set.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RUNS

BEST = sys.argv[1] if len(sys.argv) > 1 else "HPF"
OUT = RUNS / "figs"
OUT.mkdir(exist_ok=True)
SETS = ["H", "HP", "HPF", "HPFL", "HPFLD"]
COL = dict(zip(SETS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]))
lines = []


def runs(mods, split_suffix=""):
    return [RUNS / f"transformer_{mods}_T32_s{s}{split_suffix}" for s in range(3) if (RUNS / f"transformer_{mods}_T32_s{s}{split_suffix}" / "log.csv").exists()]


# ---- Fig 2: training curves ----
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
for m in SETS:
    rs = runs(m)
    if not rs:
        continue
    logs = [pd.read_csv(r / "log.csv") for r in rs]
    n = min(len(l) for l in logs)
    va = np.mean([l.val_top1.values[:n] for l in logs], 0) * 100
    tr = np.mean([l.train_acc.values[:n] for l in logs], 0) * 100
    ax[0].plot(va, color=COL[m], label=m, lw=1.4)
    ax[1].plot(tr, color=COL[m], label=m, lw=1.4)
    lines.append(f"{m}: final train acc {tr[-1]:.1f}, final val {va[-1]:.1f}, best-epoch val {max(va):.1f}, "
                 f"first epoch reaching 95% of best val {int(np.argmax(va >= 0.95 * va.max()))}")
ax[0].set_title("Validation top-1 (%)"); ax[1].set_title("Training accuracy (%)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[0].legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig2_training_curves.png"); plt.close(fig)

# ---- Fig 3: confusions ----
rs = runs(BEST)
P = pd.concat([pd.read_csv(r / "test_preds.csv").assign(seed=i) for i, r in enumerate(rs)])
names = P.drop_duplicates("label").set_index("label").gloss.to_dict()
wrong = P[P.label != P.pred]
pairs = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False).head(10)
lines.append(f"\nTop confused pairs for {BEST} (true -> predicted, count over {len(rs)} seeds):")
for (a, b), c in pairs.items():
    lines.append(f"  {names.get(a, a)} -> {names.get(b, b)}: {c}")
n = 100
cm = np.zeros((n, n))
for a, b in zip(P.label, P.pred):
    cm[a, b] += 1
cm = cm / cm.sum(1, keepdims=True).clip(1)
fig, ax = plt.subplots(figsize=(5, 4.6), dpi=200)
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(f"Row-normalised confusion ({BEST}, {len(rs)} seeds)")
fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(OUT / "fig3_confusion.png"); plt.close(fig)
per_class = P.assign(ok=(P.label == P.pred)).groupby("label").ok.mean().sort_values()
lines.append("\nWorst 10 classes (accuracy over seeds): " + ", ".join(f"{names[k]} {v*100:.0f}%" for k, v in per_class.head(10).items()))
lines.append(f"Classes at 100% accuracy: {int((per_class == 1).sum())} of {len(per_class)}")

# ---- Fig 4: per-signer ----
acc = P.assign(ok=(P.label == P.pred) * 100).groupby("signer_id").agg(acc=("ok", "mean"), videos=("video_id", "nunique")).sort_values("acc")
fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
ax.bar(range(len(acc)), acc.acc, color="#7570b3")
ax.set_xticks(range(len(acc))); ax.set_xticklabels([f"{int(s)}\n(n={int(v)})" for s, v in zip(acc.index, acc.videos)], fontsize=6)
ax.set_ylabel("top-1 (%)"); ax.set_title(f"Per-test-signer accuracy ({BEST})"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig4_per_signer.png"); plt.close(fig)

lines.append("\nPer-test-signer spread for every modality set (mean over seeds; signers with >= 5 test clips):")
for m in SETS:
    rs_m = runs(m)
    if not rs_m:
        continue
    Q = pd.concat([pd.read_csv(r / "test_preds.csv") for r in rs_m])
    g = Q.assign(ok=(Q.label == Q.pred) * 100).groupby("signer_id").agg(acc=("ok", "mean"), n=("video_id", "nunique"))
    g = g[g.n >= 5]
    lines.append(f"  {m:6s} min {g.acc.min():.1f}  max {g.acc.max():.1f}  range {g.acc.max() - g.acc.min():.1f}  std {g.acc.std():.1f}  (signers: {len(g)})")

(OUT / "figs_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("saved to", OUT)
