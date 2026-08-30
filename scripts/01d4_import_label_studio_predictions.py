import json
import os

from tasks.models import Prediction, Task


PROJECT_ID = int(os.environ.get("LS_PROJECT_ID", "2"))
MODEL_VERSION = os.environ.get("LS_MODEL_VERSION", "yolo11s-kayseri-human-v1")
INPUT_PATH = os.environ.get(
    "LS_PREDICTIONS_PATH", "/tmp/yolo11s_kayseri_human_v1_tasks.json"
)


with open(INPUT_PATH, encoding="utf-8") as source:
    prediction_tasks = json.load(source)

tasks_by_filename = {
    (task.data or {}).get("filename"): task
    for task in Task.objects.filter(project_id=PROJECT_ID)
}

updated = 0
skipped_completed = 0
missing = []
box_count = 0

for prediction_task in prediction_tasks:
    filename = prediction_task["data"]["filename"]
    task = tasks_by_filename.get(filename)
    if task is None:
        missing.append(filename)
        continue
    if task.annotations.filter(was_cancelled=False).exists():
        skipped_completed += 1
        continue

    payload = prediction_task["predictions"][0]
    result = payload.get("result", [])
    Prediction.objects.update_or_create(
        project_id=PROJECT_ID,
        task=task,
        model_version=MODEL_VERSION,
        defaults={
            "result": result,
            "score": payload.get("score", 0),
        },
    )
    task.total_predictions = Prediction.objects.filter(task=task).count()
    task.save(update_fields=["total_predictions"])
    updated += 1
    box_count += len(result)

print(f"model_version={MODEL_VERSION}")
print(f"updated_unannotated_tasks={updated}")
print(f"skipped_completed_tasks={skipped_completed}")
print(f"boxes={box_count}")
print(f"missing_tasks={len(missing)}")
