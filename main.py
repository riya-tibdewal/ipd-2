"""Main Execution Loop (Dev 4 - Assembly)"""
import time
import cv2
import config
from camera import CameraManager
from src.ai_engine import PoseDetector
from src.tracker import TrackManager
from visualization import Visualizer
from src.report import ReportGenerator

def main():
    cam = CameraManager(0)  # Change 0 to "video.mp4" for testing video files
    detector = PoseDetector()
    tracker = TrackManager()
    vis = Visualizer()

    prev_time = time.time()
    print("Starting EduAware 4-Class Pipeline... Press 'q' to exit.")

    while True:
        ret, frame = cam.get_frame()
        if not ret:
            break

        detections = detector.track_frame(frame)
        results = []

        for tid, bbox, kps in detections:
            res = tracker.process_student(tid, bbox, kps)
            results.append(res)
            vis.draw_student(frame, res)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        vis.draw_hud(frame, fps, results)

        cv2.imshow("EduAware - Behavior Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    ReportGenerator.export_csv("session_analytics_report.csv", tracker)

if __name__ == "__main__":
    main()