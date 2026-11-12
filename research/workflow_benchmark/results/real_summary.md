# Real training, VisionForge vs native

cls and seg: n = 6 seeds (37-42), VF first on 37, 39, 41 and native first on 38, 40, 42. det: n = 3 seeds (37-39), VF first on 37, 39.
Time is run wall clock. The margin for time is ±5%, checked with the 90% CI of the paired ratio.

| Recipe | Metric | VF mean ± SD | Native mean ± SD | Paired diff (95% CI) | Time VF/native, with reruns (90% CI) | Within ±5% | Time, original runs (90% CI) |
|---|---|---|---|---|---|---|---|
| ResNet-18, CIFAR-10, 30 ep | test accuracy | 0.9619 ± 0.0015 | 0.9624 ± 0.0016 | -0.0004 [-0.0012, +0.0003] | 1.026 [1.008, 1.044] | yes | 1.014 [0.982, 1.046] |
| DeepLabV3-ResNet50, Pet, 40 ep | test mIoU | 0.9799 ± 0.0007 | 0.9795 ± 0.0006 | +0.0004 [-0.0008, +0.0015] | 1.005 [0.991, 1.020] | yes | 1.057 [0.964, 1.158] |
| YOLO11s, VOC2007, 100 ep | test mAP50 | 0.7702 ± 0.0039 | 0.7702 ± 0.0039 | +0.0000 [+0.0000, +0.0000] | 0.998 [0.995, 1.000] | yes | 0.998 [0.995, 1.000] |

Reruns: both cls seed 39 runs and seg seed 37 VF fell inside a slowdown of unknown cause, about 13:11 to 15:15 on 2026-10-30.
They were repeated with the machine idle. Metrics above use the first runs. No run was dropped.
Detection metrics match to full precision, but best.pt hashes differ (det_best_pt_sha256.txt), and weight equality was not tested.
Pet mIoU ignores label 255.
