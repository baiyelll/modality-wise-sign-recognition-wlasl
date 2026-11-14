# Which Cues Matter? A Modality-Wise Study of Isolated Sign Recognition on WLASL

COMP4471 / ELEC4240 final project. We study how much each cue (hands, pose, face, lips, depth) contributes to
isolated sign recognition on WLASL100, using a modular transformer over MediaPipe Holistic landmarks and a
**signer-independent** split that we build ourselves (the official WLASL splits share signers across train/test).

## Layout

| File | Purpose |
|---|---|
| `fetch_wlasl100.py` | Downloads the 2,038 WLASL100 videos (212 MB) from the complete `gazquez/wlasl-processed` Kaggle copy into `wlasl-full/` (ranged requests, no 2.9 GB download). Resumable. |
| `config.py` | Paths. Expects `wlasl-full/` in this folder (videos + json); writes to `../data/` and `../runs/`. |
| `build_metadata.py` | Joins `WLASL_v0.3.json` + `nslt_100.json` with the videos on disk, builds the signer-independent split → `data/split.csv`. |
| `extract.py` | MediaPipe Holistic landmark extraction → `data/landmarks/<video_id>.npz`. |
| `check_landmarks.py` | Draws landmarks on one frame as a visual sanity check. |
| `data_stats.py` | Landmark-quality stats (clips flagged for >40% missing-hand frames). |
| `dataset.py` | Normalisation, augmentation, modality selection; skeleton-image renderer; RGB frame cache. |
| `model.py` | Modular transformer, BiLSTM, Pose-TGCN, skeleton-image CNN/ViT, I3D. |
| `train.py` | Trains one configuration, evaluates the best-validation checkpoint on test. |
| `run_all.py` | Runs the whole experiment plan (resumable). |
| `aggregate.py` | Builds the report tables (mean ± std over seeds). |

## Setup

Python 3.10+, a CUDA GPU for the full study (T4 is enough).

```
pip install torch torchvision numpy pandas opencv-python mediapipe==0.10.14 tqdm fvcore iopath
```

`fvcore`/`iopath` are only needed for I3D (it loads `pytorchvideo`'s `i3d_r50` through `torch.hub`, so the first run
needs internet). The CNN/ViT baselines also download ImageNet weights on first use.

## Pipeline

```
python fetch_wlasl100.py          # one-off; needs ../wlasl-processed/{WLASL_v0.3,nslt_100}.json (or copy them into wlasl-full/)
python build_metadata.py          # data/metadata.csv + data/split.csv
python extract.py                 # landmarks (resumable; --workers N, --limit N)
python check_landmarks.py 00335   # optional visual check
python data_stats.py              # flagged-clip counts for the Data section
```

## Experiments

```
python run_all.py --phase ablation   # H, H+P, H+P+F, +L, +D  x 3 seeds           (15 runs)
python run_all.py --phase split      # official (signer-dependent) split, best mods x 3 seeds
python run_all.py --phase frames     # T = 8, 16, 64, best mods, seed 0
python run_all.py --phase lstm       # BiLSTM, seed 0
python run_all.py --phase skel       # skeleton-image CNN + ViT, seed 0
python run_all.py --phase tgcn       # Pose-TGCN, seed 0
python run_all.py --phase i3d        # I3D on RGB, seed 0
python run_all.py --phase loo        # optional leave-one-out rows, seed 0
python run_all.py --phase all        # everything above except loo
python aggregate.py                  # prints tables, writes runs/summary_per_run.csv
```

- Runs whose `runs/<name>/metrics.json` exists are skipped, so an interrupted session can simply be restarted.
- "Best mods" is the ablation row with the highest mean **validation** top-1 (never test). Override with `--best HP`.
- Add `--dry` to print the commands without running them.
- A single run: `python train.py --model transformer --mods HPF --T 32 --seed 0 [--split official]`.
  Sanity checks: `--overfit` (should reach ~100% train acc) and `--shuffle_labels` (should be at chance).

Modality letters: **H** hands, **P** pose, **F** brows + eyes, **L** lips, **D** depth.

## Outputs

Each run writes `runs/<model>_<mods>_T<T>_s<seed>[_official]/` with `metrics.json` (top-1/top-5, instance and
per-class; params; train time; inference ms/clip), `config.json`, `log.csv` (per-epoch curves, for the training-behaviour
figure), `test_preds.csv` (per-video predictions, for per-signer and confusion analysis) and `best.pt`.

## Implementation notes (read before writing the report)

- **Data source.** The popular `risangbaskoro/wlasl-processed` Kaggle copy only has 1,013 of the 2,038 WLASL100 videos
  (11,980 of 21,083 overall). `gazquez/wlasl-processed` has all of them, so we use that. Its clips are already
  trimmed to the sign, so `build_metadata.py` ignores the json `frame_start/frame_end` and uses clips whole.

- **Depth (D)** is MediaPipe's own relative z-coordinate of every included landmark, *not* a monocular depth model
  (MiDaS). Without D only (x, y) are used.
- **Pose-TGCN** is our re-implementation of Li et al. (WACV'20): each keypoint is a node whose feature is its (x, y)
  trajectory over T frames, learnable adjacency, residual graph-conv blocks. It uses 67 MediaPipe nodes (25 pose +
  42 hand), not OpenPose's 55.
- **I3D** is `i3d_r50` from pytorchvideo (Kinetics-400 pretrained), new linear head, 32 frames at 160×160 crops.
- **Skeleton images** draw H+P bones of all T frames into one 224×224 image, colour-coded by time; ResNet-18 and ViT-B/16
  are ImageNet-pretrained and fine-tuned.
- Training: AdamW, cosine decay with 5 warm-up epochs, label smoothing 0.1, batch 32, 100 epochs for the landmark models;
  the baselines use the epochs/learning rates set in `run_all.py` (`EXTRA`).
- Std in the tables is the sample std (n-1) over seeds. Single-seed rows show ± 0.0 and should be labelled as such.
