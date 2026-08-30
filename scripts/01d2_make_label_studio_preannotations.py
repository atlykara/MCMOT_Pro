"""YOLO11s tahminlerini Label Studio gorev JSON'una cevirir."""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/vehicle_dataset"

COCO_TO_KAYSERI = {
    2: "car",
    5: "bus",
    7: "truck",
    3: "motorcycle",
}
KAYSERI_CLASSES = {
    "car",
    "van_minibus",
    "pickup",
    "bus",
    "truck",
    "special_vehicle",
    "motorcycle",
}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Studio icin YOLO11s on etiketleri.")
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model-version")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "label_studio_tasks.json",
    )
    args = parser.parse_args()

    image_paths = []
    for split in ("train", "val", "test"):
        image_paths.extend(sorted((DATASET_ROOT / "images" / split).glob("*.jpg")))
    if not image_paths:
        raise SystemExit("HATA: veri setinde goruntu bulunamadi.")

    model = YOLO(str(PROJECT_ROOT / args.model))
    model_names = {int(class_id): name for class_id, name in model.names.items()}
    is_kayseri_model = set(model_names.values()) == KAYSERI_CLASSES
    if is_kayseri_model:
        class_ids = sorted(model_names)
        label_for_class = model_names.get
    else:
        class_ids = sorted(COCO_TO_KAYSERI)
        label_for_class = COCO_TO_KAYSERI.get
    model_version = args.model_version or Path(args.model).stem
    print(f"Model turu: {'Kayseri 7 sinif' if is_kayseri_model else 'COCO arac siniflari'}")
    print(f"Tahmin surumu: {model_version}")
    tasks = []
    box_count = 0
    device = select_device()
    for image_number, image_path in enumerate(image_paths, 1):
        result = model.predict(
            source=str(image_path),
            classes=class_ids,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        split = image_path.parent.name
        height, width = result.orig_shape
        rectangles = []
        scores = []
        if result.boxes is not None:
            for index, (xyxy, class_id, score) in enumerate(
                zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist(), result.boxes.conf.tolist())
            ):
                label = label_for_class(class_id)
                if label is None:
                    continue
                x1, y1, x2, y2 = xyxy
                rectangles.append({
                    "id": f"{image_path.stem}-{index}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": round(float(score), 5),
                    "value": {
                        "x": 100 * x1 / width,
                        "y": 100 * y1 / height,
                        "width": 100 * (x2 - x1) / width,
                        "height": 100 * (y2 - y1) / height,
                        "rotation": 0,
                        "rectanglelabels": [label],
                    },
                })
                scores.append(float(score))
        box_count += len(rectangles)
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={split}/{image_path.name}",
                "split": split,
                "filename": image_path.name,
            },
            "predictions": [{
                "model_version": model_version,
                "score": sum(scores) / len(scores) if scores else 0,
                "result": rectangles,
            }],
        })
        if image_number % 25 == 0 or image_number == len(image_paths):
            print(f"Islenen: {image_number}/{len(image_paths)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=True), encoding="utf-8")
    print(f"Gorev: {len(tasks)}")
    print(f"On etiket kutusu: {box_count}")
    print(f"Cikti: {args.output}")


if __name__ == "__main__":
    main()
