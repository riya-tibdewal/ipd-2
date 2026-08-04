"""Main Execution & System Integration Assembly (Dev 4)"""
import time
import cv2
import config
from camera import CameraManager
from src.ai_engine import PoseDetector, FaceRecognizer, AsyncFaceWorker
from src.tracker import TrackManager
from visualization import Visualizer
from src.report import ReportGenerator

def main():
    cam = CameraManager(0)
    detector = PoseDetector()
    tracker = TrackManager()
    vis = Visualizer()
    async_worker = AsyncFaceWorker(FaceRecognizer())

    prev_time = time.time()

    print("Starting EduAware Runtime Loop. Press 'q' to exit and generate report...")

    while True:
        ret, frame = cam.get_frame()
        if not ret:
            break

        # AI Detection & Tracking
        detections = detector.track_frame(frame)
        results = []

        for tid, bbox, kps in detections:
            res = tracker.process_student(tid, bbox, kps)
            results.append(res)
            
            # Async Face Recognition Every N Frames
            if cam.frame_count % config.FACE_RECOGNITION_EVERY_N == 0:
                x1, y1, x2, y2 = bbox
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                def _update_name(target_id, name):
                    tracker.names[target_id] = name
                
                async_worker.submit_recognition(tid, crop, _update_name)

            # Draw student bounding box and skeleton
            vis.draw_student(frame, res)

        # FPS calculation & live HUD
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        vis.draw_hud(frame, fps, results)

        cv2.imshow("EduAware - Live Classroom Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Exit Cleanup & Export Session Analytics CSV
    cam.release()
    cv2.destroyAllWindows()
    ReportGenerator.export_csv("session_analytics_report.csv", tracker)

if __name__ == "__main__":
    main()