"""Same recipes as make_config.py, written directly on PyTorch / Ultralytics, no VisionForge import."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import InterpolationMode

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ROOT = Path("~/vfsmoke/datasets")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_seed(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True,
                      persistent_workers=True, worker_init_fn=worker_seed)


def run_cls(seed, epochs, out):
    seed_all(seed)
    base = [T.Resize(224), T.CenterCrop(224)]
    tail = [T.ToTensor(), T.Normalize(MEAN, STD)]
    train_tf = T.Compose(base + [T.RandomHorizontalFlip(), T.RandomRotation(10)] + tail)
    eval_tf = T.Compose(base + tail)
    d = ROOT / "cifar10-full"
    tr, va, te = (loader(ImageFolder(d / s, tf), 128, s == "train")
                  for s, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)))
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, hist = float("inf"), []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
        model.eval()
        loss = n = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                loss += F.cross_entropy(model(x), y, reduction="sum").item()
                n += y.numel()
        loss /= n
        hist.append({"epoch": ep, "val_loss": loss, "seconds": time.perf_counter() - t0})
        if loss < best:
            best = loss
            torch.save(model.state_dict(), out / "best.pth")
    train_s = time.perf_counter() - t0
    model.load_state_dict(torch.load(out / "best.pth"))
    model.eval()
    hit = n = 0
    with torch.no_grad():
        for x, y in te:
            hit += (model(x.cuda()).argmax(1).cpu() == y).sum().item()
            n += y.numel()
    return {"train_seconds": train_s, "test_accuracy": hit / n, "best_val_loss": best, "history": hist}


class Pet(Dataset):
    def __init__(self, split, train):
        d = ROOT / "pet-seg" / split
        masks = {p.stem: p for p in (d / "masks").iterdir()}
        self.pairs = [(p, masks[p.stem]) for p in sorted((d / "images").iterdir())]
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        with Image.open(ip) as im:
            image = TF.resize(im.convert("RGB"), [256, 256], interpolation=InterpolationMode.BILINEAR)
        with Image.open(mp) as mk:
            mask = TF.resize(mk.copy(), [256, 256], interpolation=InterpolationMode.NEAREST)
        if self.train and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        return TF.normalize(TF.to_tensor(image), MEAN, STD), torch.as_tensor(np.array(mask), dtype=torch.long)


def confusion(model, dl):
    cm = np.zeros((2, 2), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            p = model(x.cuda(non_blocking=True))["out"].argmax(1).cpu().numpy().ravel()
            t = y.numpy().ravel()
            keep = t != 255
            cm += np.bincount(t[keep] * 2 + p[keep], minlength=4).reshape(2, 2)
    return cm


def seg_scores(cm):
    cm = cm.astype(float)
    d, tr, gu = np.diag(cm), cm.sum(1), cm.sum(0)
    return {"miou": float((d / (tr + gu - d)).mean()), "dice": float((2 * d / (tr + gu)).mean()),
            "pixel_acc": float(d.sum() / cm.sum())}


def run_seg(seed, epochs, out):
    seed_all(seed)
    tr = loader(Pet("train", True), 16, True)
    va = loader(Pet("val", False), 16, False)
    te = loader(Pet("test", False), 16, False)
    model = deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.DEFAULT, num_classes=2, aux_loss=False).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, hist = -1.0, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x)["out"], y, ignore_index=255).backward()
            opt.step()
        sched.step()
        s = seg_scores(confusion(model, va))
        hist.append({"epoch": ep, "val_miou": s["miou"], "seconds": time.perf_counter() - t0})
        if s["miou"] > best:
            best = s["miou"]
            torch.save(model.state_dict(), out / "best.pth")
    train_s = time.perf_counter() - t0
    model.load_state_dict(torch.load(out / "best.pth"))
    return {"train_seconds": train_s, "test_metrics": seg_scores(confusion(model, te)), "best_val_miou": best, "history": hist}


def run_det(seed, epochs, out):
    from ultralytics import YOLO

    t0 = time.perf_counter()
    YOLO("yolo11s.pt").train(data="~/vfsmoke/real-v1/voc.yaml", imgsz=640, device=0, epochs=epochs, batch=16,
                             lr0=0.01, patience=0, seed=seed, deterministic=True, workers=4, optimizer="auto",
                             project=str(out), name="run", exist_ok=True, verbose=False)
    return {"train_seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["cls", "seg", "det"])
    ap.add_argument("seed", type=int)
    ap.add_argument("epochs", type=int)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"cls": run_cls, "seg": run_seg, "det": run_det}[a.task](a.seed, a.epochs, a.out)
    res.update(task=a.task, seed=a.seed, epochs=a.epochs, torch=torch.__version__)
    (a.out / "native_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
