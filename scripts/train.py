"""
Model Training and Transfer Learning CLI.
Fine-tunes a pretrained detector (YOLOv8n / YOLO11n) on target navigation classes
(Cones, Barriers, Stop Signs) using BDD100K and supplementary datasets.
"""
import argparse
from pathlib import Path
import sys
import shutil
import yaml
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.train_transfer import TransferLearningTrainer
from src.detector.dataset_adapter import BDD100KAdapter


def prepare_dataset(
    source_dir: str = "dataset/BDD100K_88mb",
    output_dir: str = "data/processed_yolo",
    max_train: int = 150,
    max_val: int = 30
) -> Path:
    """Prepares and adapts BDD100K dataset into YOLO format."""
    print(f"[*] Checking dataset preparation at {output_dir}...")
    adapter = BDD100KAdapter(output_dir=output_dir)
    data_yaml = Path(output_dir) / "data.yaml"

    src_path = Path(source_dir)
    if src_path.exists():
        print(f"[*] Processing BDD100K dataset from {source_dir}...")
        train_count = adapter.convert_dataset_ninja_bdd100k(
            str(src_path / "train"), split="train", max_samples=max_train
        )
        val_count = adapter.convert_dataset_ninja_bdd100k(
            str(src_path / "val"), split="val", max_samples=max_val
        )
        test_count = adapter.convert_dataset_ninja_bdd100k(
            str(src_path / "test"), split="test", max_samples=max_val
        )
        print(f"[OK] Converted {train_count} train, {val_count} val, {test_count} test images.")
        yaml_path = adapter.generate_yolo_yaml()
    elif Path("data/processed_robotics/data.yaml").exists():
        yaml_path = Path("data/processed_robotics/data.yaml")
        print(f"[*] Using existing robotics dataset at {yaml_path}")
    else:
        raise FileNotFoundError(f"Neither {source_dir} nor data/processed_robotics found.")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train / Fine-Tune Detector for Robotic Navigation")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (e.g. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml dataset definition")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-weights", type=str, default="weights/robotics_best.pt", help="Destination path for best checkpoint")
    args = parser.parse_args()

    print("=================================================================")
    print("      ROBOTICS OBJECT DETECTION: TRANSFER LEARNING TRAINER      ")
    print("=================================================================\n")
    print(f"[*] Base Model: {args.model}")
    print(f"[*] Device: {args.device.upper()}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 1. Dataset verification / preparation
    if args.data and Path(args.data).exists():
        data_yaml = Path(args.data)
    else:
        data_yaml = prepare_dataset()

    print(f"[*] Dataset Configuration: {data_yaml}")

    # 2. Transfer Learning Trainer
    trainer = TransferLearningTrainer(
        base_model_path=args.model,
        data_yaml_path=str(data_yaml),
        device=args.device
    )

    # 3. Fine-tuning execution
    res = trainer.fine_tune(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        freeze_backbone_layers=args.freeze,
        project_dir="runs/train",
        experiment_name="robotics_fine_tune"
    )

    best_w = Path(res["best_weights"])
    print(f"\n[OK] Training completed! Best checkpoint: {best_w}")

    # 4. Copy to target weights directory
    out_target = Path(args.output_weights)
    out_target.parent.mkdir(parents=True, exist_ok=True)
    if best_w.exists():
        shutil.copy(best_w, out_target)
        print(f"[OK] Saved production weights to: {out_target}")

    # 5. Validation metrics
    print("\n[*] Running final validation metrics...")
    val_metrics = trainer.validate(str(out_target if out_target.exists() else best_w))
    print(f"  -> mAP50     : {val_metrics['mAP50']:.4f}")
    print(f"  -> mAP50-95  : {val_metrics['mAP50_95']:.4f}")
    print(f"  -> Precision : {val_metrics['precision']:.4f}")
    print(f"  -> Recall    : {val_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
