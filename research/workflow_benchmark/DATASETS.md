# Public datasets for the workflow benchmark

Selection reviewed on 2026-10-23. These are established evaluation datasets;
their use does not make VisionForge or its default architectures state of the
art. The experiment must match the native workflow before measuring tool effects.
USK-COFFEE remains a useful domain-specific external validation case.

## Staged corpus

| Task | Dataset and primary source | Proposed use | Access and current limitation |
|---|---|---|---|
| Classification | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | First CPU parity pilot, then a separately frozen full-data recipe | Official binary archive: 162 MB as listed by the authors. The source page asks for citation but does not state a dataset-wide license; do not infer an MIT/CC license or redistribute images with the code. |
| Detection | [COCO 2017](https://cocodataset.org/#download) | Matched torchvision or Ultralytics detector; AP with the official evaluation semantics | Authors list train images 18 GB, validation images 1 GB, train/validation annotations 241 MB. Requires separate download authorization and matching category IDs, crowd handling and evaluator. |
| Segmentation | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | Binary foreground/background segmentation with the ambiguous trimap boundary ignored | Authors list approximately 800 MB; CC BY-SA 4.0, image copyright remains with original owners. Preserve distributed test membership and derive validation only from training data. |
| Anomaly | [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Normal-only training per category; image AUROC initially | CC BY-NC-SA 4.0. Official page currently reports unavailable downloads. Existing authorized copies can be used after checking provenance; never silently replace the source. |
| Anomaly, harder extension | [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | Later robustness extension, not the first compatibility target | CC BY-NC-SA 4.0; form-gated download, partly private test labels and official evaluation server. Current image-only metrics do not establish pixel-localization performance. |
| Regression, candidate | [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) | Age regression with identity-disjoint splits and MAE/RMSE; protocol still to freeze | Non-commercial research only; authors require requesting the ZIP password and forbid redistribution. No download, contact or acceptance of terms is automated. Select an alternative if these conditions are unsuitable. |

COCO download sizes are recorded in the project's own
[download inventory](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm).
Image use follows the [COCO terms](https://cocodataset.org/#termsofuse), not the
license of its API implementation. Access and terms can change; recheck at acquisition.
This table is a proposed expansion, not evidence that the adapters already exist.
Classification and segmentation have implemented paired native baselines in this
directory. Segmentation is verified only on synthetic CPU fixtures; the
Oxford-IIIT Pet trimap mapping (1 pet -> 1, 2 background -> 0, 3 border -> 255
ignored) is fixed in `trimap_labels.py` and ADR-149, but the dataset has not been
acquired, split or run. Its acquisition, validation split, architecture and
budget are separate steps that need explicit authorization.

## First CPU pilot

`prepare_cifar10.py` consumes an already downloaded **binary** archive. It neither
downloads data nor imports torch. It verifies the MD5 published by the authors,
records SHA-256, reads fixed records without extracting archive paths, and creates
lossless PNGs in a new ImageFolder directory. Existing outputs are refused.

The default seed is 37. Each of the ten classes contributes 32 training,
8 validation and 8 test images: 320/80/80 total. Training and validation are
disjoint selections from the official 50,000-image training pool; test images
come only from the official 10,000-image test pool. A subset is **not** a standard
full CIFAR-10 accuracy result. `manifest.json` records each original split/index,
class, generated PNG hash, decoded pixel hash, RNG and NumPy version. Save it
with the paired report. Inspect the normal image audit before executing a pair;
findings must remain visible rather than selecting another subset after seeing scores.

```powershell
# Run from an isolated checkout. Use the existing environment; do not sync it.
$python = 'C:\path\to\VisionForge\.venv\Scripts\python.exe' # existing environment
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $python -m research.workflow_benchmark.prepare_cifar10 datasets/cifar-10-binary.tar.gz datasets/cifar10-pilot
```

Create one local, CPU ResNet18 state dictionary with ten outputs, without
pretrained downloads, and pass that same file to both implementations:

```python
import torch
from torch import nn
from torchvision import models

torch.random.default_generator.manual_seed(37)
with torch.device("cpu"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
torch.save(model.state_dict(), "datasets/cifar10-initial.pth")
```

```powershell
& $python -m research.workflow_benchmark.paired_classification datasets/cifar10-pilot datasets/cifar10-initial.pth outputs/cifar10-pair --seed 37 --epochs 2 --batch-size 16
```

Use below-normal process priority and one CPU thread while another training job
is active. CUDA must remain hidden before importing torch; no pretrained weights
are downloaded. CPU/I/O contention can still affect the existing job. Run one
small pair at a time and stop only the pilot if resource use becomes a problem.
Concurrent-workload timings are excluded from comparative performance results.
Passing means numerical parity for this recipe, not competitive model quality,
training speed, workflow usability or a completed scientific benchmark.

## Full study gate

Freeze a separate recipe with the complete intended corpus, validation policy,
preprocessing, initialization, training budget, output scope and metrics before
observing test scores. Use the same splits and seeds in every matched condition;
prespecify at least three seeds and report every run, including failures. Three
seeds are an initial replication budget, not a power calculation or a guarantee
of statistical significance. Add native baselines for each additional task before
collecting comparisons. Measure GPU runtime only on an otherwise available GPU.

Source images, archives and weights stay in ignored local directories. Publish
scripts, protocols and permitted evidence; obtain explicit authorization before
committing generated research results or redistributing any dataset material.

## Additional detection parity corpus

| Task | Dataset and primary source | Recipe and limitation |
|---|---|---|
| Detection | [Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) | Local VOCtrainval_06-Nov-2007.tar and VOCtest_06-Nov-2007.tar; official train/val/test lists, 20 alphabetical classes, difficult excluded. `prepare_voc.py` writes YOLO labels and archive hashes. SSDLite parity uses shared all-points mAP@50, not official VOC2007 11-point AP; real-corpus execution is separate from synthetic CPU checks. |
