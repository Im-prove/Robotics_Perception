"""
Generates augmented training & validation dataset for Cones, Barriers, and Stop Signs,
and performs transfer learning (freezing backbone layers) to produce fine-tuned robotics weights.
"""
from pathlib import Path
import random
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from scripts.generate_sample_data import (
    create_realistic_road_scene,
    draw_metric_cone,
    draw_metric_barrier,
    draw_metric_stop_sign
)
from src.detector.train_transfer import TransferLearningTrainer


def generate_robotics_dataset(output_dir: Path, num_train: int = 40, num_val: int = 10):
    """Generate a balanced, metric-accurate dataset for transfer learning."""
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open("configs/camera_config.yaml", "r") as f:
        cam_cfg = yaml.safe_load(f)["camera"]
    intrinsics = cam_cfg["intrinsics"]
    intrinsics["image_width"] = cam_cfg["image_width"]
    intrinsics["image_height"] = cam_cfg["image_height"]

    w = 1280
    h = 720

    splits = [("train", num_train, train_img_dir, train_lbl_dir),
              ("val", num_val, val_img_dir, val_lbl_dir)]

    for split_name, count, img_dir, lbl_dir in splits:
        print(f"[*] Generating {count} {split_name} samples...")
        for i in range(count):
            img = create_realistic_road_scene(width=w, height=h)
            labels = []

            # Add random cones (0 to 3)
            num_cones = random.randint(1, 3)
            for _ in range(num_cones):
                dist = random.uniform(1.2, 9.0)
                offset = random.uniform(-1.5, 1.5)
                lbl = draw_metric_cone(img, distance_z=dist, offset_x=offset, intrinsics=intrinsics)
                labels.append((0, lbl["box2d"])) # class 0 = Cone

            # Add barrier (0 or 1)
            if random.random() > 0.4:
                dist = random.uniform(2.5, 10.0)
                offset = random.uniform(-1.8, 1.8)
                lbl = draw_metric_barrier(img, distance_z=dist, offset_x=offset, intrinsics=intrinsics)
                labels.append((1, lbl["box2d"])) # class 1 = Barrier

            # Add stop sign (0 or 1)
            if random.random() > 0.3:
                dist = random.uniform(2.5, 12.0)
                offset = random.uniform(0.8, 2.2) # usually on right side of road
                lbl = draw_metric_stop_sign(img, distance_z=dist, offset_x=offset, intrinsics=intrinsics)
                labels.append((2, lbl["box2d"])) # class 2 = Stop Sign

            # Save image
            img_filename = f"scene_{split_name}_{i:03d}.jpg"
            cv2.imwrite(str(img_dir / img_filename), img)

            # Save label in YOLO format: class_id cx cy w h (normalized)
            lbl_filename = f"scene_{split_name}_{i:03d}.txt"
            with open(lbl_dir / lbl_filename, "w", encoding="utf-8") as f_lbl:
                for cls_id, box in labels:
                    x1, y1 = box["x1"], box["y1"]
                    x2, y2 = box["x2"], box["y2"]
                    bw = max(0.001, (x2 - x1) / w)
                    bh = max(0.001, (y2 - y1) / h)
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    # Clip to [0, 1]
                    cx = min(1.0, max(0.0, cx))
                    cy = min(1.0, max(0.0, cy))
                    bw = min(1.0, max(0.0, bw))
                    bh = min(1.0, max(0.0, bh))
                    f_lbl.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Generate data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Cone", 1: "Barrier", 2: "Stop Sign"},
        "nc": 3
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"[OK] Dataset prepared at {output_dir} with data.yaml at {yaml_path}")
    return yaml_path


def main():
    dataset_dir = Path("data/processed_robotics")
    yaml_path = generate_robotics_dataset(dataset_dir, num_train=35, num_val=10)

    # Transfer learning: fine-tune pretrained yolov8n.pt with frozen backbone
    print("\n[*] Starting transfer learning fine-tuning with frozen backbone...")
    trainer = TransferLearningTrainer(
        base_model_path="yolov8n.pt",
        data_yaml_path=str(yaml_path),
        device="cpu"
    )

    # Train for 5 fast epochs to adapt weights to the 3 robotics classes
    res = trainer.fine_tune(
        epochs=5,
        imgsz=640,
        batch=8,
        lr0=0.01,
        freeze_backbone_layers=8,
        project_dir="runs/train",
        experiment_name="fine_tuned_robotics"
    )

    best_w = res["best_weights"]
    print(f"\n[OK] Transfer Learning Fine-Tuning Completed! Best weights saved at: {best_w}")

    # Copy best weights to weights/robotics_best.pt for easy reference
    import shutil
    target_p = Path("weights/robotics_best.pt")
    target_p.parent.mkdir(parents=True, exist_ok=True)
    if Path(best_w).exists():
        shutil.copy(best_w, target_p)
        print(f"[OK] Copied weights to {target_p}")


if __name__ == "__main__":
    main()
