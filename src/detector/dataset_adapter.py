"""
BDD100K Dataset Adapter and Robotics Data Pipeline.
Parses BDD100K JSON annotations, filters target navigation classes (cone, barrier, stop sign),
and converts to standard YOLO format for fine-tuning.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import cv2
import numpy as np
import yaml


class BDD100KAdapter:
    """Adapter for BDD100K detection labels and dataset preparation."""

    TARGET_MAPPING = {
        "cone": 0,
        "traffic cone": 0,
        "barrier": 1,
        "traffic barrier": 1,
        "jersey barrier": 1,
        "stop sign": 2,
        "stop_sign": 2,
    }

    CLASS_NAMES = ["Cone", "Barrier", "Stop Sign"]

    def __init__(self, output_dir: str = "data/processed_yolo"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

    def convert_bdd100k_json(
        self,
        json_path: str,
        image_src_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> int:
        """
        Convert BDD100K JSON format to standard YOLO annotation format.
        BDD100K JSON schema:
        [
          {
            "name": "0000f77c-62c2a288.jpg",
            "labels": [
              {
                "category": "traffic sign",
                "box2d": {"x1": 1000.0, "y1": 200.0, "x2": 1040.0, "y2": 240.0},
                "attributes": {"trafficLightColor": "none", ...}
              }
            ]
          }
        ]
        """
        split_img_dir = self.images_dir / split
        split_lbl_dir = self.labels_dir / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        converted_count = 0

        for item in data:
            if max_samples and converted_count >= max_samples:
                break

            img_name = item.get("name")
            labels = item.get("labels", [])

            valid_boxes = []

            for label in labels:
                category = label.get("category", "").lower()
                attributes = label.get("attributes", {})
                box2d = label.get("box2d")

                if not box2d:
                    continue

                # Determine if label matches target classes
                matched_id = None
                if "cone" in category:
                    matched_id = 0
                elif "barrier" in category:
                    matched_id = 1
                elif "stop" in category or attributes.get("sub_category") == "stop sign":
                    matched_id = 2
                elif category == "traffic sign" and "stop" in str(label).lower():
                    matched_id = 2

                if matched_id is not None:
                    valid_boxes.append((matched_id, box2d))

            if not valid_boxes:
                continue

            # Copy image and write label file
            src_img_path = Path(image_src_dir) / img_name
            if not src_img_path.exists():
                continue

            img = cv2.imread(str(src_img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            dest_img_path = split_img_dir / img_name
            cv2.imwrite(str(dest_img_path), img)

            lbl_filename = Path(img_name).stem + ".txt"
            dest_lbl_path = split_lbl_dir / lbl_filename

            with open(dest_lbl_path, "w", encoding="utf-8") as f_lbl:
                for cls_id, box in valid_boxes:
                    x1, y1 = float(box["x1"]), float(box["y1"])
                    x2, y2 = float(box["x2"]), float(box["y2"])

                    # Normalize coordinates to [0, 1]
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    f_lbl.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            converted_count += 1

        self.generate_yolo_yaml()
        return converted_count

    def convert_dataset_ninja_bdd100k(
        self,
        ninja_split_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> int:
        """
        Convert Supervisely / Dataset Ninja BDD100K structure (train/ann and train/img)
        to YOLO dataset format.
        """
        split_img_dir = self.images_dir / split
        split_lbl_dir = self.labels_dir / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        ann_dir = Path(ninja_split_dir) / "ann"
        img_src_dir = Path(ninja_split_dir) / "img"

        if not ann_dir.exists() or not img_src_dir.exists():
            return 0

        ann_files = list(ann_dir.glob("*.json"))
        converted = 0

        for ann_file in ann_files:
            if max_samples and converted >= max_samples:
                break

            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            objects = data.get("objects", [])
            valid_boxes = []

            for obj in objects:
                title = obj.get("classTitle", "").lower()
                pts = obj.get("points", {}).get("exterior", [])
                if len(pts) != 2:
                    continue

                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])

                matched_id = None
                if "cone" in title:
                    matched_id = 0
                elif "barrier" in title:
                    matched_id = 1
                elif "stop" in title or "sign" in title:
                    matched_id = 2

                if matched_id is not None:
                    valid_boxes.append((matched_id, x1, y1, x2, y2))

            if not valid_boxes:
                continue

            # Find matching image
            img_name = ann_file.stem
            img_path = img_src_dir / img_name
            if not img_path.exists():
                # Try without .jpg suffix or with .jpg
                alt_name = img_name.replace(".jpg", "") + ".jpg"
                img_path = img_src_dir / alt_name
                if not img_path.exists():
                    continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            dest_img_path = split_img_dir / img_path.name
            cv2.imwrite(str(dest_img_path), img)

            lbl_filename = img_path.stem + ".txt"
            with open(split_lbl_dir / lbl_filename, "w", encoding="utf-8") as f_lbl:
                for cls_id, x1, y1, x2, y2 in valid_boxes:
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f_lbl.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            converted += 1

        self.generate_yolo_yaml()
        return converted

    def generate_yolo_yaml(self) -> Path:
        """Create the data.yaml configuration file for YOLO fine-tuning."""
        yaml_content = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(self.CLASS_NAMES)},
            "nc": len(self.CLASS_NAMES)
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return yaml_path
