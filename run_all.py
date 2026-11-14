"""Runs the full experiment plan by calling train.py. Resumable: a run whose metrics.json exists is skipped.

  python run_all.py --phase ablation          # 5 mod sets x 3 seeds
  python run_all.py --phase split             # signer-dependent (official split), best mods, 3 seeds
  python run_all.py --phase frames            # T = 8, 16, 64, best mods, seed 0
  python run_all.py --phase lstm              # BiLSTM, best mods, seed 0
  python run_all.py --phase skel              # skeleton-image CNN + ViT, seed 0
  python run_all.py --phase tgcn              # Pose-TGCN (re-implementation), seed 0
  python run_all.py --phase i3d               # I3D RGB, seed 0
  python run_all.py --phase loo               # optional leave-one-out (seed 0)
  python run_all.py --phase all [--dry]

"Best mods" = the ablation row with the highest mean VALIDATION top-1 (never test), or --best XYZ.
"""
import argparse, subprocess, sys
import pandas as pd
from config import RUNS

MODS = ["H", "HP", "HPF", "HPFL", "HPFLD"]
SEEDS = [0, 1, 2]
# per-baseline hyper-parameters (the transformer/LSTM use train.py defaults: 100 ep, lr 3e-4)
EXTRA = {
    "tgcn": ["--epochs", "100", "--lr", "1e-3"],
    "cnn": ["--epochs", "40", "--lr", "3e-4", "--workers", "2"],
    "vit": ["--epochs", "30", "--lr", "5e-5", "--amp", "--workers", "2"],
    "i3d": ["--epochs", "30", "--lr", "1e-4", "--bs", "8", "--amp", "--workers", "4"],
}


def run_name(model, mods, T, seed, split):
    return f"{model}_{mods}_T{T}_s{seed}" + ("_official" if split == "official" else "")


def launch(model, mods, T, seed, split="si", dry=False):
    name = run_name(model, mods, T, seed, split)
    if (RUNS / name / "metrics.json").exists():
        print(f"[skip] {name}")
        return
    cmd = [sys.executable, "train.py", "--model", model, "--mods", mods, "--T", str(T),
           "--seed", str(seed), "--split", split, *EXTRA.get(model, [])]
    print("[run ]", " ".join(cmd[1:]), flush=True)
    if not dry and subprocess.call(cmd) != 0:
        print(f"[FAIL] {name} (continuing)")


def best_mods(forced=None):
    if forced:
        return forced
    rows = []
    for m in MODS:
        for s in SEEDS:
            f = RUNS / run_name("transformer", m, 32, s, "si") / "metrics.json"
            if f.exists():
                rows.append((m, pd.read_json(f, typ="series")["val_top1"]))
    if not rows:
        sys.exit("No ablation runs yet; run --phase ablation first or pass --best")
    d = pd.DataFrame(rows, columns=["m", "v"]).groupby("m").v.agg(["mean", "count"])
    print("validation top-1 by mod set:\n", d.to_string())
    if (d["count"] < len(SEEDS)).any():
        print("WARNING: ablation incomplete, best-mods choice is provisional")
    return d["mean"].idxmax()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--best", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    P = a.phase
    todo = lambda *n: P in ("all", *n)

    if todo("ablation"):
        for m in MODS:
            for s in SEEDS:
                launch("transformer", m, 32, s, dry=a.dry)
    if todo("split", "frames", "lstm", "loo"):
        best = best_mods(a.best)
        print("best mods (by val):", best)
    if todo("split"):
        for s in SEEDS:
            launch("transformer", best, 32, s, "official", a.dry)
    if todo("frames"):
        for T in (8, 16, 64):
            launch("transformer", best, T, 0, dry=a.dry)
    if todo("lstm"):
        launch("lstm", best, 32, 0, dry=a.dry)
    if todo("skel"):
        launch("cnn", "HP", 32, 0, dry=a.dry)
        launch("vit", "HP", 32, 0, dry=a.dry)
    if todo("tgcn"):
        launch("tgcn", "HP", 32, 0, dry=a.dry)
    if todo("i3d"):
        launch("i3d", "RGB", 32, 0, dry=a.dry)
    if P == "loo":  # not part of "all": optional, seed 0 only
        full = "HPFLD"
        for drop in "FLD":
            launch("transformer", full.replace(drop, ""), 32, 0, dry=a.dry)
        launch("transformer", "".join(c for c in full if c != "P"), 32, 0, dry=a.dry)
