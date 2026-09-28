
### mean SSIMULACRA2 FR (higher = closer to the reference)

| regime / class | NERVE-fid4x-down | NERVE-gan2x | NERVE-rel2x | RealESRGAN-4x-down | bicubic-cg | lanczos-mps | REFERENCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicubic photo (5) | 37.31 | 38.84 | 53.99 | 14.97 | 48.09 | 46.28 | 100.00 |
| bicubic graphic (4) | 57.85 | 61.36 | 69.32 | 40.25 | 63.17 | 53.53 | 100.00 |
| lanczos photo (5) | 45.62 | 43.05 | 59.72 | 20.94 | 52.67 | 46.46 | 100.00 |
| lanczos graphic (4) | 67.17 | 67.70 | 68.95 | 50.11 | 63.07 | 48.08 | 100.00 |
| jpeg photo (5) | 13.11 | 13.72 | 11.74 | -3.82 | 13.51 | 12.70 | 100.00 |
| jpeg graphic (4) | 45.43 | 48.26 | 44.65 | 34.16 | 44.18 | 39.12 | 100.00 |

### mean gradient (detail; reference in the REFERENCE column)

| regime / class | NERVE-fid4x-down | NERVE-gan2x | NERVE-rel2x | RealESRGAN-4x-down | bicubic-cg | lanczos-mps | REFERENCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicubic photo (5) | 4.36 | 4.93 | 5.23 | 4.60 | 3.71 | 4.14 | 6.59 |
| bicubic graphic (4) | 1.88 | 1.96 | 1.78 | 2.21 | 1.67 | 2.07 | 1.71 |
| lanczos photo (5) | 4.31 | 4.98 | 5.64 | 4.54 | 3.91 | 4.37 | 6.59 |
| lanczos graphic (4) | 1.91 | 2.02 | 2.07 | 2.23 | 1.80 | 2.18 | 1.71 |
| jpeg photo (5) | 3.56 | 4.11 | 3.97 | 3.96 | 3.18 | 3.50 | 6.59 |
| jpeg graphic (4) | 1.63 | 1.91 | 2.01 | 2.11 | 1.74 | 2.06 | 1.71 |

### FidelityGuard DC worst 64 px tile (higher = less invented)

| regime / class | NERVE-fid4x-down | NERVE-gan2x | NERVE-rel2x | RealESRGAN-4x-down | bicubic-cg | lanczos-mps | REFERENCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicubic photo (5) | 0.8685 | 0.8538 | 0.9709 | 0.7681 | 0.9783 | 0.9797 | 0.9016 |
| bicubic graphic (4) | 0.9679 | 0.9675 | 0.9968 | 0.9385 | 0.9954 | 0.9972 | 0.9805 |
| lanczos photo (5) | 0.8516 | 0.8374 | 0.9703 | 0.7519 | 0.9766 | 0.9786 | 0.9371 |
| lanczos graphic (4) | 0.9719 | 0.9612 | 0.9966 | 0.9347 | 0.9944 | 0.9967 | 0.9937 |
| jpeg photo (5) | 0.8972 | 0.8431 | 0.9812 | 0.7465 | 0.9850 | 0.9864 | 0.6662 |
| jpeg graphic (4) | 0.9259 | 0.9258 | 0.9961 | 0.9030 | 0.9966 | 0.9978 | 0.8874 |

### jpeg regime, per image FR

| image | NERVE-fid4x-down | NERVE-gan2x | NERVE-rel2x | RealESRGAN-4x-down | bicubic-cg | lanczos-mps |
|---|---:|---:|---:|---:|---:|---:|
| photo-a | 25.44 | 24.76 | 20.96 | 6.16 | 22.04 | 21.17 |
| graphic-a | 37.23 | 39.89 | 37.53 | 24.48 | 37.06 | 34.38 |
| photo-b | 8.35 | 11.72 | 6.45 | -8.33 | 10.12 | 9.32 |
| graphic-b | 46.33 | 52.32 | 44.99 | 32.65 | 42.87 | 38.67 |
| photo-c | -0.01 | 0.50 | 0.68 | -15.22 | 1.51 | 0.25 |
| graphic-c | 40.85 | 42.89 | 41.99 | 31.64 | 41.92 | 37.64 |
| photo-d | 8.92 | 4.50 | 8.07 | -13.49 | 9.68 | 8.80 |
| graphic-d | 57.30 | 57.95 | 54.08 | 47.88 | 54.88 | 45.79 |
| photo-e | 22.87 | 27.12 | 22.55 | 11.76 | 24.18 | 23.96 |

### route decision: `.fidelity` ×4 + downsample vs native 2x_NERVE_OTF_gan, per cell

- bicubic: fid4x-down wins 2/9, gan2x wins 7/9; mean FR(fid4x-down) − FR(gan2x) = -2.41
- lanczos: fid4x-down wins 6/9, gan2x wins 3/9; mean FR(fid4x-down) − FR(gan2x) = +1.19
- jpeg: fid4x-down wins 2/9, gan2x wins 7/9; mean FR(fid4x-down) − FR(gan2x) = -1.60
- all 27 cells: fid4x-down wins 10/27, mean Δ -0.94
