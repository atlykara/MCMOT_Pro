"""Etiketlenmis Kayseri verisiyle YOLO11s iki asamali fine-tuning."""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_labels(dataset_root: Path) -> dict[str, int]:
    counts = {}
    errors = []
    for split in ("train", "val", "test"):
        images = sorted((dataset_root / "images" / split).glob("*.*"))
        labels_dir = dataset_root / "labels" / split
        counts[split] = len(images)
        if split in {"train", "val"} and not images:
            errors.append(f"{split}: goruntu yok")
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"etiket eksik: {label_path.relative_to(PROJECT_ROOT)}")
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                try:
                    class_id = int(values[0])
                    coords = [float(value) for value in values[1:]]
                except (ValueError, IndexError):
                    errors.append(f"gecersiz satir: {label_path.name}:{line_number}")
                    continue
                if len(coords) != 4 or not 0 <= class_id <= 6 or any(not 0 <= value <= 1 for value in coords):
                    errors.append(f"gecersiz YOLO etiketi: {label_path.name}:{line_number}")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        extra = f"\n- ... ve {len(errors) - 20} hata daha" if len(errors) > 20 else ""
        raise SystemExit(f"HATA: egitim baslatilmadi.\n{preview}{extra}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11s Kayseri arac modeli fine-tuning.")
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument(
        "--dataset",
        default="data/vehicle_dataset",
        help="dataset.yaml, images/ ve labels/ iceren veri kumesi klasoru",
    )
    parser.add_argument("--run-name", default="yolo11s_kayseri")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset_root = PROJECT_ROOT / args.dataset
    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        raise SystemExit(f"HATA: dataset.yaml bulunamadi: {dataset_yaml}")
    counts = validate_labels(dataset_root)
    device = select_device()
    print(f"Veri: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"Device: {device}")
    if device == "cpu":
        print("UYARI: CPU egitimi cok yavas olur; CUDA GPU veya calisan MPS ortami onerilir.")

    project = PROJECT_ROOT / "runs/vehicle_detector"
    stage1 = YOLO(str(PROJECT_ROOT / args.model))
    stage1.train(
        data=str(dataset_yaml),
        epochs=args.stage1_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        freeze=10,
        project=str(project),
        name=f"{args.run_name}_stage1",
        exist_ok=True,
        seed=42,
    )

    stage1_best = project / f"{args.run_name}_stage1/weights/best.pt"
    stage2 = YOLO(str(stage1_best))
    stage2.train(
        data=str(dataset_yaml),
        epochs=args.stage2_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        lr0=0.001,
        project=str(project),
        name=f"{args.run_name}_stage2",
        exist_ok=True,
        seed=42,
    )

    best = project / f"{args.run_name}_stage2/weights/best.pt"
    deployed = PROJECT_ROOT / f"models/{args.run_name}.pt"
    shutil.copy2(best, deployed)
    metrics = YOLO(str(best)).val(data=str(dataset_yaml), split="val", device=device)
    print(f"Secilen agirlik: {deployed}")
    print(f"Dogrulama mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
