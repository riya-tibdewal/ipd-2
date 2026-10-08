"""Create an evaluation-safe model for the optimized web pipeline.

Unlike the legacy trainer, this keeps frames from one video together in either
training or validation, avoiding adjacent-frame data leakage.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

from src.ai_engine import PoseDetector
from src.optimized_metrics import FEATURE_NAMES, extract_optimized_metrics

LABEL_MAP = {"WRITING": 0, "TALKING": 1, "USING_PHONE": 2, "SLEEPING": 3}
INVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
ROOT = Path(__file__).resolve().parent


def extract_dataset(video_root: Path = ROOT / "data" / "training_videos") -> pd.DataFrame:
    detector, rows = PoseDetector("yolov8n-pose.pt"), []
    for video_path in sorted(video_root.rglob("*")):
        if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        label = video_path.parent.name.upper()
        if label not in LABEL_MAP:
            continue
        capture, last_keypoints = cv2.VideoCapture(str(video_path)), {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for track_id, bbox, keypoints, has_phone, has_notebook in detector.track_frame(frame):
                metrics = extract_optimized_metrics(keypoints, bbox, last_keypoints.get(track_id))
                metrics.has_phone, metrics.has_notebook = float(has_phone), float(has_notebook)
                last_keypoints[track_id] = keypoints.copy()
                row = {name: getattr(metrics, name) for name in FEATURE_NAMES}
                row.update(target_label=LABEL_MAP[label], source_video=str(video_path))
                rows.append(row)
        capture.release()
    if not rows:
        raise RuntimeError(f"No labelled videos found under {video_root}")
    dataset = pd.DataFrame(rows)
    dataset.to_csv(ROOT / "data" / "optimized_pose_dataset.csv", index=False)
    return dataset


def train(dataset: pd.DataFrame, output: Path = ROOT / "models" / "optimized_behavior_models.joblib") -> None:
    groups = dataset["source_video"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_indices, test_indices = next(splitter.split(dataset, dataset.target_label, groups))
    training, testing = dataset.iloc[train_indices], dataset.iloc[test_indices]
    model = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(training[list(FEATURE_NAMES)], training.target_label)
    predictions = model.predict(testing[list(FEATURE_NAMES)])
    print(classification_report(testing.target_label, predictions, labels=sorted(INVERSE_LABEL_MAP),
          target_names=[INVERSE_LABEL_MAP[index] for index in sorted(INVERSE_LABEL_MAP)], zero_division=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tier2": model, "tier1": None, "label_map": INVERSE_LABEL_MAP,
                 "feature_names": FEATURE_NAMES, "training_videos": sorted(groups.unique())}, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    train(extract_dataset())
