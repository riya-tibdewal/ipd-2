# EduAware implementation changes

## Existing pipeline preserved

The original `main.py`, legacy tracker, detector, visualizer, and legacy
training script remain available for the existing VS Code/OpenCV workflow.
The new functionality is additive.

## Added web application

| Area | Added file(s) | Purpose |
| --- | --- | --- |
| Backend API | `backend/main.py` | FastAPI service for video upload, starting a session, polling progress, retrieving results, and downloading reports. |
| Frontend | `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` | Browser dashboard for uploading a video, viewing progress, reviewing per-track results, and downloading CSV output. |
| Web dependencies | `requirements-web.txt` | Lists FastAPI, Uvicorn, and multipart upload support. |
| Runtime outputs | `runtime/uploads/`, `runtime/reports/` | Created at runtime; ignored by Git. |

## Added optimized analysis path

| File | Improvement |
| --- | --- |
| `src/optimized_metrics.py` | Provides one consistent 14-feature extraction contract, restores `vertical_extension`, `leg_visibility`, and `seated_cue`, and corrects torso-lean scaling. |
| `src/optimized_tracker.py` | Uses time-based (two-second) smoothing instead of a fixed frame count, retains reportable session state, and clears transient data for stale tracks. |
| `src/session_reporting.py` | Exports a report compatible with the optimized tracker: final label, confidence, average attention, observations, and label counts. |
| `src/video_session.py` | Runs video analysis without OpenCV UI, allowing the backend and future clients to reuse the same pipeline. |

## Added safer retraining workflow

`train_optimized_model.py` builds a new dataset using the optimized feature
extractor and trains a Random Forest using whole-video train/validation splits.
This avoids leakage from adjacent video frames appearing in both splits.

It creates:

- `data/optimized_pose_dataset.csv`
- `models/optimized_behavior_models.joblib`

The optimized runtime selects that model automatically once it exists;
otherwise it uses the current `models/behavior_models.joblib`.

## Verification completed

- New Python files passed syntax parsing.
- FastAPI routes loaded successfully.
- The optimized tracker loaded the current model and produced a prediction from
  a synthetic pose input.

## Run the web demo

```powershell
cd "C:\tech\FINAL IPD\ipd"
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

## Recommended next step

After reviewing the UI, retrain using the optimized feature path:

```powershell
python train_optimized_model.py
```

Review the printed validation metrics before relying on the new model.
