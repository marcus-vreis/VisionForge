# gui-v1, VisionForge through the browser vs native

n = 3 seeds (37, 38, 39). Margins frozen in gui-v1/PROTOCOL.md: time ±5% (90% CI), quality ±0.005 (95% CI).
Failing the quality margin at n = 3 means inconclusive, not worse.

| Recipe | Metric | Paired diff (95% CI) | Quality within | Time GUI/native (90% CI) | Per seed | Time within | Click to end/native (90% CI) |
|---|---|---|---|---|---|---|---|
| ResNet-18, CIFAR-10, 30 ep | test accuracy | -0.0005 [-0.0026, +0.0016] | yes | 1.168 [1.017, 1.341] | 1.124, 1.283, 1.104 | no | 1.160 [1.009, 1.334] |
| DeepLabV3-ResNet50, Pet, 40 ep | test mIoU | +0.0007 [-0.0034, +0.0049] | yes | 1.120 [1.084, 1.157] | 1.145, 1.103, 1.112 | no | 1.114 [1.077, 1.152] |
