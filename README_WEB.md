# EduAware web demo

The original VS Code/OpenCV application remains unchanged. This added web
layer processes an uploaded video using the optimized pipeline and returns a
per-track summary plus a downloadable CSV.

## Start

From the project root:

```powershell
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`, upload an MP4/AVI/MOV/MKV file, then choose
**Start analysis**. The API is available under `/api` and the generated CSV is
available from the dashboard when processing completes.

## Retrain after feature changes

Use the new trainer rather than the legacy one:

```powershell
python train_optimized_model.py
```

It generates `data/optimized_pose_dataset.csv`, holds entire videos out of
validation to avoid adjacent-frame leakage, and writes
`models/optimized_behavior_models.joblib`. Review the printed per-class
precision/recall before replacing a production model.
