# gui-v2, VisionForge through the browser vs native, after the animation fix

n = 3 seeds (37, 38, 39). Margins frozen in gui-v2/PROTOCOL.md: time ±5% (90% CI), quality ±0.005 (95% CI).
Failing the quality margin at n = 3 means inconclusive, not worse.

| Recipe | Metric | Paired diff (95% CI) | Quality within | Time GUI/native (90% CI) | Per seed | Time within | Click to end/native (90% CI) |
|---|---|---|---|---|---|---|---|
| ResNet-18, CIFAR-10, 30 ep | test accuracy | -0.0005 [-0.0026, +0.0016] | yes | 1.047 [1.006, 1.089] | 1.041, 1.026, 1.074 | no | 1.039 [0.998, 1.082] |
| DeepLabV3-ResNet50, Pet, 40 ep | test mIoU | +0.0000 [-0.0026, +0.0027] | yes | 1.017 [1.014, 1.020] | 1.018, 1.015, 1.018 | yes | 1.012 [1.009, 1.015] |
