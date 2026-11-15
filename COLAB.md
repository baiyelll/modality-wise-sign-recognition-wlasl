# Running the experiments on Colab (two people, two T4s)

Landmarks are already extracted, so Colab only does **training** (no MediaPipe needed). One person = one Colab
session on its own Google account (free Colab allows one GPU session per account, and the free GPU quota is not
guaranteed, so a paid tier is a safe fallback).

## Work split

The only dependency is that "best modality set" comes from the ablation, so the person who runs the ablation also runs
the phases that need it. The other person runs the baselines, which use fixed inputs and can start immediately.

| | Runs | Phases | Rough time (plan estimate) |
|---|---|---|---|
| **Person A** | modality ablation (15) + signer-dependent (3) + frame count (3) + BiLSTM (1) | `ablation`, then `split`, `frames`, `lstm` | ~4-5 h |
| **Person B** | Pose-TGCN, skeleton CNN + ViT, I3D (4) | `tgcn`, `skel`, `i3d` | ~2.5-5 h |

If one finishes early, run the optional leave-one-out rows (`--phase loo`) for Table 2.
Timings are estimates from the project plan; time one `--epochs 2` run first and extrapolate (x50).

## Files to put on Google Drive (once, either person)

- `colab_data.zip` (332 MB, made in `Documents/`): 2,038 landmark files, `split.csv`, and the two WLASL json files.
  Put it at `MyDrive/wlasl/colab_data.zip`. Both people use the **same** zip so both train on the identical split.
- The code comes from the GitHub repo (`git clone`); the repo must have the latest commit pushed first.

## Setup cells (both people; run top to bottom)

```python
# 1. GPU + Drive
!nvidia-smi -L
from google.colab import drive; drive.mount('/content/drive')
```
```python
# 2. code + data
!git clone https://github.com/baiyelll/modality-wise-sign-recognition-wlasl.git /content/repo
!mkdir -p /content/work && unzip -q -o /content/drive/MyDrive/wlasl/colab_data.zip -d /content/work
%cd /content/repo
!pip -q install fvcore iopath        # only needed for I3D (Person B)
import os
os.environ["WLASL_ROOT"] = "/content/work"            # data/ and runs/ live here
os.environ["WLASL_RAW"]  = "/content/work/wlasl-full"  # videos + json (only needed for I3D)
```
```python
# 3. restore earlier results after a disconnect, then keep syncing runs to Drive every 5 min
#    (use runs_A for Person A and runs_B for Person B; best.pt checkpoints are not synced, not needed)
ME = "A"   # or "B"
!mkdir -p /content/work/runs /content/drive/MyDrive/wlasl/runs_{ME}
!rsync -a /content/drive/MyDrive/wlasl/runs_{ME}/ /content/work/runs/
import subprocess
subprocess.Popen(f"while true; do rsync -a --exclude best.pt /content/work/runs/ /content/drive/MyDrive/wlasl/runs_{ME}/; sleep 300; done", shell=True)
```
```python
# 4. smoke test (~1 min): prints the device (must be cuda), params, a speed per epoch. Then delete it.
!python train.py --mods HP --epochs 2 --seed 99
!rm -rf /content/work/runs/transformer_HP_T32_s99 /content/work/runs/results.csv
```

## Person A

```python
!python run_all.py --phase ablation          # 15 runs, resumable
!python run_all.py --phase split             # best mods chosen on VALIDATION accuracy, then official split x3 seeds
!python run_all.py --phase frames
!python run_all.py --phase lstm
```
Use `--dry` first if you want to see the commands. If the session dies, repeat cells 1-3 and re-run: finished runs are
skipped (a run in progress restarts from epoch 0).

## Person B

```python
# I3D needs the videos (212 MB, ~minutes). The json files are already in WLASL_RAW from the zip.
!python fetch_wlasl100.py
!python run_all.py --phase tgcn
!python run_all.py --phase skel
!python run_all.py --phase i3d               # first call also builds a ~6 GB frame cache, then trains
```

## Merging and tables (on the laptop, or any one machine)

1. Download `runs_A` and `runs_B` from Drive (or share the folders) and copy **all run folders** into one `runs/`
   directory (`Documents/runs/`). Folder names are unique per configuration, so nothing collides.
2. `python aggregate.py` prints Tables 2 and 3, the frame-count table, the signer-dependent gap and per-signer accuracy.
3. Give `runs/summary_per_run.csv`, the aggregate output, and `README.md` to whoever updates the report (see its Part A).

## Gotchas
- `WLASL_ROOT` / `WLASL_RAW` must be set in every new Colab session (cell 2). Without them the code looks in the folder above the repo.
- `runs/results.csv` is appended to by each run; the report tables do not use it (they read each run's `metrics.json`).
- Do not run `build_metadata.py` on Colab: it rewrites `split.csv`; ship the one from the zip so both people share one split.
- ViT and I3D download pretrained weights on first use (internet is available on Colab).
