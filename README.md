# Which Cues Matter? A Modality-Wise Study of Isolated Sign Recognition on WLASL

COMP4471 / ELEC4240 final project. We study how much each cue (hands, pose, face, lips, depth) contributes to
isolated sign recognition on WLASL100, using a modular transformer over MediaPipe Holistic landmarks and a
**signer-independent** split that we build ourselves (the official WLASL splits share signers across train/test).

This README has two parts: **A. Project status and report-update guide** (read this first if you are updating the
LaTeX report) and **B. How to run everything**.

---

# A. Project status and report-update guide

## A0. Rules for whoever updates the report
- Fill every `\tbd{...}` **only** from files in `../runs/` (run `python aggregate.py`) or `../data/`. Never invent or
  "round to something plausible". If a result does not exist yet, leave the `\tbd{}` and say so.
- Numbers in the current draft that are not marked `\tbd` (52.4 / 61.8 / +9.4, "44% to 78%" per signer, "1,764 videos",
  "11 test signers", "18 ms per frame", "about 12 minutes per run", "about 3.2M parameters", "13% not downloaded") are
  **milestone/mock values from an earlier, different data set**. All must be replaced by numbers from the final runs.
- Model selection uses **validation** accuracy only; test numbers are reported, never used to choose anything.

## A1. Status (as of 2026-11-15)

| Step | State |
|---|---|
| Videos (all 2,038 WLASL100) | **Done**, in `wlasl-full/videos/` |
| `build_metadata.py` (split) | **Done**, `../data/split.csv` |
| Landmark extraction | **Done**, 2,038/2,038 clips in `../data/landmarks/` (0 failures) |
| Landmark-quality stats | **Done** (see A2) |
| Colab setup (`COLAB.md`, `../colab_data.zip`) | **Prepared**, not yet used |
| All training experiments (26 runs) | **NOT RUN YET.** Code is written and smoke-tested on synthetic data only |
| Figures 2 (training curves), 3 (confusions), per-signer plot | **No plotting scripts exist yet** (data for them is produced by `train.py`) |
| Webcam demo / text-to-speech | Not implemented (draft §5.7 says to move it to future work if unfinished) |

## A2. Facts already established (safe to put in the report)

**Data source.** `risangbaskoro/wlasl-processed` (the common Kaggle copy) has only 11,980 of WLASL's 21,083 videos and
only 1,013 of the 2,038 WLASL100 videos (1,025 missing). We therefore use `gazquez/wlasl-processed`, which contains all
2,038 (verified: all present, real files, median ~94 KB, 219 MB total). Videos are 256x256, median 59 frames, 25 fps.
These clips are already trimmed to the sign, whereas the json `frame_start/frame_end` index the original untrimmed
videos (131 of 2,038 had `frame_start` beyond the end of the file), so `build_metadata.py` ignores the json ranges and
uses every clip whole.

**Split (signer-independent, built by `build_metadata.py`, seed 0).**

| Split | Videos | Signers | Glosses |
|---|---|---|---|
| Train | 1,402 | 72 | 100 |
| Validation | 243 | 10 | 100 |
| Test | 393 | 15 | 100 |
| Total | 2,038 | 97 | 100 |

No signer appears in two splits (asserted in the script); every gloss appears in train, val and test. Compute the median
videos per gloss from `../data/split.csv` (`groupby("gloss").size().median()`); mean is 20.4. (Not yet computed.)

**Landmark quality** (`python data_stats.py`, 2,038 clips, 126,802 frames, 12-203 frames per clip, mean 62.3):
- Both hands missing in >40% of frames: **770 clips (37.8%)** (train 575, val 125, test 70). None removed.
- "Either hand missing >40%" flags 1,576 (77.3%) which is uninformative (one-handed signs leave the other hand
  undetected), so the report uses the **both-hands** definition. Mean missing-frame rate: left hand 64%, right hand 36%,
  pose 0.2%, face 0.5%.
- **Upscaling test:** on 20 random flagged clips, upscaling frames 2x / 3x before MediaPipe did not change the missing
  rate (both-hands missing: 0.532 original, 0.534 at 2x, 0.533 at 3x). So low resolution is *not* the cause; the likely
  cause (hands out of frame / resting / occluded / blurred) was **not verified by looking at frames**. Say so, or check.
- No clip is dropped anywhere in the pipeline; missing landmarks are linearly interpolated along time (all-missing -> 0).

**Extraction cost** (for §5.7; replace the draft's "18 ms per frame on a laptop CPU"): single worker on the team
laptop measured ~69 ms/frame (3-clip test); the full run used 11 workers on 12 cores: 61.5 min wall, 34.4 frames/s
overall (319 ms/frame per core under load). Transformer inference ms/clip is recorded per run in `metrics.json`
(`infer_ms_per_clip`; measured at batch size 1 on whatever device trained it).

## A3. Where each report TBD comes from

| Report item | Source |
|---|---|
| Abstract numbers, §5.1 Table 2 (modality ablation), conclusion | `python aggregate.py` -> "Table 2" (mean ± sample std, 3 seeds) |
| Is a gain within seed noise? | compare the difference to the std in Table 2 |
| §5.1 leave-one-out row (optional) | `python run_all.py --phase loo` (seed 0 only; label it single-seed) |
| §5.2 training behaviour, Fig. 2 | `../runs/<run>/log.csv` (epoch, train_loss, train_acc, val_top1, val_loss). **Verify** the draft claims ("converge within ~60 epochs", "hands-only overfits more") against these curves before keeping them. Needs a plotting script |
| §5.3 frame count | aggregate.py "Frame count" table (seed 0; T=32 row is the ablation's seed 0) |
| §5.4 Table 3 | aggregate.py "Table 3". Rows for BiLSTM, CNN, ViT, I3D, Pose-TGCN are **single seed**. Keep official-split I3D 65.89 / Pose-TGCN 55.43 (from the WLASL paper) in separate labelled rows |
| §5.5 signer-dependent gap | `python run_all.py --phase split` then aggregate.py "Signer-dependent vs signer-independent" |
| §5.5 per-signer spread | aggregate.py "Per-test-signer top-1" (15 test signers, not 11). Whether F/L/D narrow the spread needs the same table per modality set (only the best set is printed; extend if needed) |
| §5.6 error analysis, Fig. 3, failure cases | `../runs/<run>/test_preds.csv` (video_id, signer_id, gloss, label, pred, top5). Sign pairs fixed by face/lips and minimal-pair failures need manual inspection |
| §5.7 efficiency | A2 above + `infer_ms_per_clip`, `train_time_s` in `metrics.json` |
| §3 Table 1, Data, Preprocessing | A2 above |

## A4. Draft text that does NOT match the code (fix the text or the code before submitting)
1. **Depth.** §2/§4.1 say depth is sampled from a monocular depth model (MiDaS). The code's `D` is MediaPipe's own
   relative z-coordinate of every included landmark. Reword the paper (recommended) or implement MiDaS.
2. **Coordinates.** §4.1 says hands/pose have (x, y, z) (126-d / 75-d). Without `D` the code uses **(x, y) only**
   (hands 84-d, pose 50-d, brows+eyes 104-d, lips 80-d); `D` adds z (126 / 75 / 156 / 120). Figure 1 stream sizes must
   match the configuration being described. Face = 52 landmarks (eyes + eyebrows), lips = 40 landmarks.
3. **Normalisation (Eq. 1).** The paper divides by a per-frame shoulder width w_t; the code divides by the **median
   shoulder width over the clip** (one scalar per clip) and subtracts the per-frame shoulder midpoint. Resampling to T
   frames is uniform index sampling (not interpolation); interpolation is only used for missing detections.
4. **Augmentation.** Paper: "rotation, scaling, temporal cropping, landmark dropout". Code: rotation ±15°, scale
   0.9-1.1, temporal crop keeping 80-100% of the clip, and dropout that zeroes **whole frames** (p = 0.1) per stream,
   not individual landmarks.
5. **Training details to state:** AdamW (weight decay 0.05), lr 3e-4, 5 warm-up epochs then cosine decay, label smoothing 0.1, grad-clip 1.0, batch 32, 100 epochs; checkpoint chosen by best validation top-1.
6. **Parameter count / run time.** "About 3.2M parameters" and "12 minutes on a GTX 1080 Ti" are unverified; the full
   H+P+F+L+D transformer has ~3.58M parameters. Take both from `metrics.json` (`n_params`, `train_time_s`) and name the real GPU.
7. **Baselines.** Pose-TGCN is our re-implementation (67 MediaPipe nodes, not OpenPose's 55). I3D is pytorchvideo
   `i3d_r50` (Kinetics-400), not the Inception-I3D of the WLASL paper, fine-tuned on 32 frames at 160x160. State this in §4.4.
8. **Limitations.** The "13% of videos could not be downloaded" sentence no longer applies (100% obtained). Add: hand landmarks missing in many clips (A2), single-seed baselines, depth is MediaPipe z, Kaggle mirror choice.
9. **Team contributions** need to reflect the actual division of work.

## A5. Pitfalls
- Landmark files and `split.csv` live in `../data/`, run outputs in `../runs/` (the **parent** of this folder), not in the repo.
- A system Python with an old pandas + numpy 2 prints `AttributeError: _ARRAY_API not found` (harmless noise) or fails
  to import pandas; `python -m pip install -U pandas` fixes it.
- `build_metadata.py` rewrites `split.csv`; re-running it with different split settings changes Table 1 and invalidates finished runs.
- Windows/Colab: the T4 machine only needs `wlasl-full/` (re-run `fetch_wlasl100.py`), then `build_metadata.py` and `extract.py`, or copy `../data/` across.

---

# B. How to run everything

**For two people on Google Colab T4s, see [COLAB.md](COLAB.md)** (work split, setup cells, merging results). Set `WLASL_ROOT` / `WLASL_RAW` to relocate data and runs.

## Layout

| File | Purpose |
|---|---|
| `fetch_wlasl100.py` | Downloads the 2,038 WLASL100 videos (212 MB) from the complete `gazquez/wlasl-processed` Kaggle copy into `wlasl-full/` (ranged requests, no 2.9 GB download). Resumable. |
| `mp_constants.py` | Frozen MediaPipe index lists (face regions, bone connections) so training needs no mediapipe. |
| `config.py` | Paths. Expects `wlasl-full/` in this folder (videos + json); writes to `../data/` and `../runs/`. |
| `build_metadata.py` | Joins `WLASL_v0.3.json` + `nslt_100.json` with the videos on disk, builds the signer-independent split -> `data/split.csv`. |
| `extract.py` | MediaPipe Holistic landmark extraction -> `data/landmarks/<video_id>.npz`. |
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

Modality letters: **H** hands, **P** pose, **F** brows + eyes, **L** lips, **D** depth (MediaPipe z).
Estimated cost on a T4: ablation ~3 h, signer-dependent ~36 min, frames ~36 min, BiLSTM ~12 min, CNN+ViT 1-2 h, I3D + Pose-TGCN 1-3 h.

## Outputs

Each run writes `runs/<model>_<mods>_T<T>_s<seed>[_official]/` with `metrics.json` (top-1/top-5, instance and
per-class; params; train time; inference ms/clip), `config.json`, `log.csv` (per-epoch curves, for the training-behaviour
figure), `test_preds.csv` (per-video predictions, for per-signer and confusion analysis) and `best.pt`.
Aggregates: `runs/summary_per_run.csv`, `runs/extract_log.csv`.

## Implementation notes

- **Pose-TGCN** is our re-implementation of Li et al. (WACV'20): each keypoint is a node whose feature is its (x, y)
  trajectory over T frames, learnable adjacency, residual graph-conv blocks. 67 nodes (25 pose + 42 hand).
- **I3D** is `i3d_r50` from pytorchvideo (Kinetics-400 pretrained), new linear head, 32 frames at 160x160 crops.
- **Skeleton images** draw H+P bones of all T frames into one 224x224 image, colour-coded by time; ResNet-18 and ViT-B/16
  are ImageNet-pretrained and fine-tuned. (Renderer only tested on synthetic data; eyeball a few real images first.)
- Baseline hyper-parameters are set in `run_all.py` (`EXTRA`); landmark models use `train.py` defaults.
- Std in the tables is the sample std (n-1) over seeds. Single-seed rows show ± 0.0 and must be labelled as single-seed.
