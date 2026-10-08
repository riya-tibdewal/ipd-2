# # """Main Execution Loop (EduAware Classroom Analytics)"""
# import sys
# import os
# import time
# import cv2

# # Explicitly add root directory to Python path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import config
# from src.ai_engine import PoseDetector
# from src.tracker import TrackManager
# from src.visualization import Visualizer
# from src.report import ReportGenerator


# class CameraManager:
#     """Video Feed & Capture Manager"""
#     def __init__(self, source=0):
#         self.cap = cv2.VideoCapture(source)
#         self.frame_count = 0

#     def get_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             self.frame_count += 1
#         return ret, frame

#     def release(self):
#         self.cap.release()


# def main():
#     # Source: Use video file or 0 for live webcam
#     video_path = "data/training_videos/USING_PHONE/using_phone2.mp4"
#     source = video_path if os.path.exists(video_path) else 0

#     cam = CameraManager(source)
#     detector = PoseDetector()
#     tracker = TrackManager()
#     vis = Visualizer()

#     prev_time = time.time()
#     print(f"Starting EduAware 4-Class Pipeline on source: {source}... Press 'q' to exit.")

#     while True:
#         ret, frame = cam.get_frame()
#         if not ret:
#             print("Video stream finished or frame unreadable.")
#             break

#         # 1. Detect keypoints, phone presence, and notebook presence
#         detections = detector.track_frame(frame)
#         results = []

#         # 2. Track, score, and visualize each student
#         for tid, bbox, kps, has_phone, has_notebook in detections:
#             res = tracker.process_student(tid, bbox, kps, has_phone, has_notebook)
#             results.append(res)
#             vis.draw_student(frame, res)

#         # 3. Calculate FPS & Overlay HUD
#         curr_time = time.time()
#         fps = 1.0 / max(curr_time - prev_time, 1e-5)
#         prev_time = curr_time

#         vis.draw_hud(frame, fps, results)

#         # 4. Show Window
#         cv2.imshow("EduAware - Behavior Analytics", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 5. Cleanup & Export Session Analytics
#     cam.release()
#     cv2.destroyAllWindows()
#     ReportGenerator.export_csv("session_analytics_report.csv", tracker)
#     print("Session complete. Analytics report saved to 'session_analytics_report.csv'.")


# if __name__ == "__main__":
#     main()

"""Main Execution Loop (EduAware Classroom Analytics)"""
import sys
import os
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.ai_engine import PoseDetector
from src.tracker import TrackManager
from src.visualization import Visualizer
from src.report import ReportGenerator

class CameraManager:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame

    def release(self):
        self.cap.release()

def main():
    video_path = "data/train_sleeping.mp4"
    source = video_path if os.path.exists(video_path) else 0

    cam = CameraManager(source)
    detector = PoseDetector()
    tracker = TrackManager()
    vis = Visualizer()

    prev_time = time.time()
    print(f"Starting EduAware 4-Class Pipeline on source: {source}... Press 'q' to exit.")

    while True:
        ret, frame = cam.get_frame()
        if not ret:
            break

        detections = detector.track_frame(frame)
        results = []

        for tid, bbox, kps, has_phone, has_notebook in detections:
            res = tracker.process_student(tid, bbox, kps, has_phone, has_notebook)
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
    print("Session complete. Analytics report saved to 'session_analytics_report.csv'.")

if __name__ == "__main__":
    main()