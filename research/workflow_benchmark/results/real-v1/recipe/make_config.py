"""Write the VisionForge YAML of one realistic-recipe run (cls | seg | det)."""
import sys
from pathlib import Path

import yaml


def config(task: str, seed: int, epochs: int, out: Path) -> dict:
    output = {k: str(out / k.split("_")[0]) for k in ("models_dir", "graphics_dir", "logs_dir", "reports_dir")}
    if task == "cls":
        return {
            "schema_version": 1, "name": f"real_cls_s{seed}", "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 10, "pretrained": True},
            "training": {"learning_rate": 3e-4, "epochs": epochs, "batch_size": 128, "early_stopping_patience": 0,
                         "optimizer": "adamw", "weight_decay": 0.01, "seed": seed, "deterministic": True,
                         "mixed_precision": False, "scheduler": {"kind": "cosine"}},
            "data": {"base_dir": "~/vfsmoke/datasets/cifar10-full", "train_dir": "train", "val_dir": "val",
                     "test_dir": "test", "num_workers": 4, "pin_memory": True,
                     "transforms": {"image_size": 224, "augment": True, "horizontal_flip": True, "rotation_degrees": 10,
                                    "color_jitter": False}},
            "output": output, "device": {"kind": "cuda"},
        }
    if task == "seg":
        return {
            "schema_version": 1, "name": f"real_seg_s{seed}", "task": "segmentation",
            "model": {"name": "deeplabv3_resnet50", "num_classes": 2, "pretrained": True},
            "training": {"learning_rate": 3e-4, "epochs": epochs, "batch_size": 16, "early_stopping_patience": 0,
                         "optimizer": "adamw", "weight_decay": 0.01, "loss": "cross_entropy", "seed": seed,
                         "deterministic": True, "scheduler": {"kind": "cosine"}},
            "data": {"base_dir": "~/vfsmoke/datasets/pet-seg", "images_subdir": "images", "masks_subdir": "masks",
                     "train_dir": "train", "val_dir": "val", "test_dir": "test", "image_size": 256, "ignore_index": 255,
                     "num_workers": 4, "pin_memory": True,
                     "transforms": {"augment": True, "horizontal_flip": True, "rotation_degrees": 0, "color_jitter": False}},
            "output": output, "device": {"kind": "cuda"},
        }
    return {
        "schema_version": 1, "name": f"real_det_s{seed}", "task": "detection",
        "model": {"backend": "ultralytics", "name": "yolo11s", "num_classes": 20, "pretrained": True},
        "data": {"data_yaml": "~/vfsmoke/real-v1/voc.yaml", "image_size": 640},
        "training": {"epochs": epochs, "batch_size": 16, "learning_rate": 0.01, "patience": 0, "seed": seed,
                     "deterministic": True, "workers": 4},
        "output": output, "device": {"kind": "cuda"},
    }


if __name__ == "__main__":
    task, seed, epochs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config(task, seed, epochs, out), sort_keys=False), encoding="utf-8")
