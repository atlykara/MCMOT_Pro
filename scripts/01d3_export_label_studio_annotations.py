"""Label Studio insan onaylarini surumlu bir YOLO veri kumesine aktarir."""

import argparse
import csv
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data/vehicle_dataset"
CLASS_NAMES = (
    "car",
    "van_minibus",
    "pickup",
    "bus",
    "truck",
    "special_vehicle",
    "motorcycle",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def latest_annotations(connection: sqlite3.Connection, project_id: int) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return connection.execute(
        """
        SELECT annotation_id, task_id, created_at, updated_at, data, result
        FROM (
            SELECT
                a.id AS annotation_id,
                a.task_id,
                a.created_at,
                a.updated_at,
                t.data,
                a.result,
                ROW_NUMBER() OVER (
                    PARTITION BY a.task_id ORDER BY a.updated_at DESC, a.id DESC
                ) AS row_number
            FROM task_completion AS a
            JOIN task AS t ON t.id = a.task_id
            WHERE a.project_id = ? AND a.was_cancelled = 0
        )
        WHERE row_number = 1
        ORDER BY task_id
        """,
        (project_id,),
    ).fetchall()


def rectangle_to_yolo(item: dict) -> tuple[int, float, float, float, float]:
    if item.get("type") != "rectanglelabels":
        raise ValueError(f"desteklenmeyen sonuc tipi: {item.get('type')}")
    value = item["value"]
    rotation = float(value.get("rotation", 0))
    if rotation != 0:
        raise ValueError("donmus kutular YOLO bicimine dogrudan aktarilamaz")
    labels = value.get("rectanglelabels", [])
    if len(labels) != 1 or labels[0] not in CLASS_IDS:
        raise ValueError(f"bilinmeyen sinif: {labels}")

    x = float(value["x"]) / 100.0
    y = float(value["y"]) / 100.0
    width = float(value["width"]) / 100.0
    height = float(value["height"]) / 100.0
    if width <= 0 or height <= 0:
        raise ValueError("kutu genisligi ve yuksekligi pozitif olmali")

    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + width))
    y2 = max(0.0, min(1.0, y + height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("kutu goruntu sinirlari disinda")
    return (
        CLASS_IDS[labels[0]],
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    )


def write_dataset_yaml(output_root: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_root / "dataset.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio insan etiketlerini YOLO veri kumesine aktarir."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / "Desktop/label-studio-data/label_studio.sqlite3",
    )
    parser.add_argument("--project-id", type=int, default=2)
    parser.add_argument("--version", default="human_v1")
    args = parser.parse_args()

    output_root = SOURCE_ROOT / "iterations" / args.version
    if output_root.exists():
        raise SystemExit(
            f"HATA: veri surumu zaten var: {output_root}\n"
            "Yeni bir --version adi kullanin; eski egitim verisi sessizce ezilmez."
        )

    with sqlite3.connect(args.db) as connection:
        rows = latest_annotations(connection, args.project_id)
    if not rows:
        raise SystemExit("HATA: gonderilmis insan etiketi bulunamadi.")

    manifest_rows = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()

    for row in rows:
        data = json.loads(row["data"])
        results = json.loads(row["result"])
        filename = data["filename"]
        split = data["split"]
        if split not in {"train", "val"}:
            raise SystemExit(f"HATA: ara egitimde desteklenmeyen split: {split}")

        source_image = SOURCE_ROOT / "images" / split / filename
        if not source_image.exists():
            raise SystemExit(f"HATA: kaynak goruntu yok: {source_image}")
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_dir / filename)

        yolo_lines = []
        for item in results:
            try:
                class_id, cx, cy, width, height = rectangle_to_yolo(item)
            except (KeyError, TypeError, ValueError) as error:
                raise SystemExit(
                    f"HATA: task={row['task_id']} dosya={filename}: {error}"
                ) from error
            yolo_lines.append(
                f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
            )
            class_counts[CLASS_NAMES[class_id]] += 1

        (label_dir / f"{Path(filename).stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )
        camera_id = filename.split("_", 1)[0]
        split_counts[split] += 1
        camera_counts[camera_id] += 1
        manifest_rows.append(
            {
                "task_id": row["task_id"],
                "annotation_id": row["annotation_id"],
                "updated_at": row["updated_at"],
                "camera_id": camera_id,
                "split": split,
                "filename": filename,
                "box_count": len(yolo_lines),
            }
        )

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_dataset_yaml(output_root)

    summary = {
        "version": args.version,
        "label_studio_project_id": args.project_id,
        "annotations": len(rows),
        "boxes": sum(class_counts.values()),
        "splits": dict(sorted(split_counts.items())),
        "cameras": dict(sorted(camera_counts.items())),
        "classes": {name: class_counts[name] for name in CLASS_NAMES},
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Veri kumesi: {output_root}")


if __name__ == "__main__":
    main()
