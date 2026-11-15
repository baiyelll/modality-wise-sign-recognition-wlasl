# Final numbers for the report (all from ../runs, 26 runs, signer-independent test unless stated)

Hardware: Colab T4. Mean ± sample std over 3 seeds for the transformer; baselines are single seed (label them so).

## Data (replace draft values)
- 2,038 videos, 100 glosses, 97 signers; median videos per gloss = 20 (mean 20.4, min 18, max 40).
- Split: train 1,402 (72 signers) / val 243 (10) / test 393 (15). Test has 15 signers, not 11.
- Both hands missing in >40% of frames: 770 clips (37.8%), none removed.
- Videos: all 2,038 obtained from the gazquez/wlasl-processed Kaggle mirror (the more common risangbaskoro copy has 1,013/2,038).

## Table 2: modality ablation (test inst. top-1, mean ± std, 3 seeds)
| Mods | top-1 | cls top-1 | top-5 | val |
|---|---|---|---|---|
| H | 70.2 ± 0.4 | 74.8 ± 0.8 | 86.1 ± 0.4 | 81.5 |
| HP | 71.0 ± 1.0 | 75.6 ± 1.4 | 87.4 ± 0.4 | 82.4 |
| HPF | 71.4 ± 2.1 | 75.1 ± 2.0 | 86.3 ± 2.1 | 82.9 |
| HPFL | 70.4 ± 0.6 | 75.2 ± 1.4 | 86.3 ± 0.8 | 82.7 |
| HPFLD | 65.6 ± 0.5 | 70.8 ± 0.8 | 85.4 ± 0.4 | 78.1 |
Best set on validation: HPF. H, HP, HPF, HPFL differ by less than the seed std (HPF std 2.1); only adding MediaPipe z (HPFLD) is a clear change (-5.8 vs HPF, val also drops 82.9 -> 78.1, all 3 seeds). D = MediaPipe relative z, not MiDaS depth.

## Frame count (HPF, seed 0)
T=8: 70.2 | T=16: 71.0 | T=32: 73.8 | T=64: 69.2 (single seed; T=32 differs from the 3-seed mean 71.4 because it is seed 0 only).
Inference ms/clip: 1.5 / 1.9 / 2.1 / 1.7. Train s: 96 / 101 / 116 / 161.

## Table 3: model comparison (signer-independent test)
| Model | top-1 | cls top-1 | top-5 | seeds |
|---|---|---|---|---|
| Transformer (HPF) | 71.4 ± 2.1 | 75.1 | 86.3 | 3 |
| BiLSTM (HPF) | 60.8 | 64.8 | 81.9 | 1 |
| Pose-TGCN (HP) | 53.9 | 58.8 | 77.6 | 1 |
| ResNet-18 skeleton image (HP) | 51.7 | 56.3 | 73.0 | 1 |
| I3D (RGB) | 50.9 | 54.7 | 74.8 | 1 |
| ViT-B/16 skeleton image (HP) | 47.8 | 49.9 | 71.8 | 1 |
Efficiency (params / train time / inference ms per clip on T4): transformer HPF 3.45M / 116 s / 2.1; BiLSTM 2.94M / 96 s / 2.6; TGCN 1.19M / 75 s / 2.3; CNN 11.2M / 470 s / 3.1; ViT 85.9M / 580 s / 17.2; I3D 27.4M / 3,592 s / 55.3. Full HPFLD transformer: 3.58M params.
I3D val 65.8 vs test 50.9 (large gap; train acc 100%). Official-split numbers quoted from the WLASL paper (I3D 65.89, Pose-TGCN 55.43) must stay in separately labelled rows.

## Signer-dependent vs signer-independent (HPF, 3 seeds)
Official split: 79.7 ± 1.6 top-1 (top-5 93.3 ± 0.8); signer-independent: 71.4 ± 2.1. Gap = +8.3 points.

## Per-signer (HPF, 15 test signers, 393 clips)
Range 33.3% (signer 117, 2 clips) to 100%. Signers with >= 5 clips: HPF 41.7 -> 100 (std 19.0). Signer 11 alone has 164/393 test clips (67.3%). Range per set (signers with >=5 clips): H 54.6, HP 62.5, HPF 58.3, HPFL 62.5, HPFLD 55.6 points, so adding F/L/D does NOT narrow the per-signer spread.

## Training behaviour (Fig. 2, runs/figs/fig2_training_curves.png)
All sets reach 100% train accuracy by ~epoch 50; val top-1 plateaus around epoch 40-45 (first epoch at 95% of best val: H 35, HP 42, HPF 43, HPFL 41, HPFLD 44) and stays flat to epoch 100. Final val: H 80.0, HP 80.9, HPF 82.2, HPFL 81.3, HPFLD 76.8. All models overfit (train 100% vs val ~80%). The draft claim "hands-only overfits more" is NOT supported: H has the same 100% train acc and a similar val curve; the visible difference is HPFLD (slower, lower).

## Error analysis (Fig. 3/4, runs/figs/figs_summary.txt)
HPF, 3 seeds: 34 of 100 classes at 100%; worst: same 8%, tell 20%, give 23%, corn 27%, accident 28%. Most frequent confusions: accident->wrong, tell->who, woman->full, last->before, right->year, same->time. Whether these are face/lips-fixed pairs needs manual inspection.

## Still not final
- Leave-one-out rows were not run (optional).
- Baselines and frame-count rows are single-seed.
- Hand-missing cause (hands out of frame) not verified by viewing frames.
