"""CSV & Analytics Summary Generator (Dev 3)"""
import csv
from typing import Dict
from src.tracker import TrackManager

class ReportGenerator:
    @staticmethod
    def export_csv(filepath: str, tracker: TrackManager):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Track ID", "Student Name", "Average Attention (%)", "Final Stable Label"])
            for tid, state in tracker.states.items():
                name = tracker.names.get(tid, f"Student-{tid}")
                writer.writerow([tid, name, f"{state.ema_attention:.2f}", state.current_label])
        print(f"Analytics report successfully saved to {filepath}")