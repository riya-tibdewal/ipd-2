"""CSV export compatible with the optimized tracking state."""
import csv
from pathlib import Path

from src.optimized_tracker import OptimizedTrackManager


def export_session_csv(path: Path, tracker: OptimizedTrackManager) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "track_id", "student_name", "final_behavior", "final_confidence",
            "average_attention", "observations", "behavior_counts",
        ))
        writer.writeheader()
        for item in tracker.session_summary():
            writer.writerow({"track_id": item["track_id"], "student_name": item["name"],
                             "final_behavior": item["final_label"], "final_confidence": item["final_confidence"],
                             "average_attention": item["average_attention"], "observations": item["observations"],
                             "behavior_counts": item["label_counts"]})
