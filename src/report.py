"""CSV Analytics Summary Generator (Dev 3)"""
import csv
from src.tracker import TrackManager

class ReportGenerator:
    @staticmethod
    def export_csv(filepath: str, tracker: TrackManager):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Track ID", "Student Name", "Attentiveness Status", "Final Behavior Class", "Average Attention Score (%)"])
            for tid, state in tracker.states.items():
                name = f"Student-{tid}"
                status = "ATTENTIVE" if state.current_label == "WRITING" else "NON-ATTENTIVE"
                writer.writerow([tid, name, status, state.current_label, f"{state.ema_attention:.2f}"])
        print(f"Analytics report saved to {filepath}")