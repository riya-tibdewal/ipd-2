"""YOLOv8 Pose & InsightFace Thread (Dev 1)"""
import concurrent.futures
import pickle
import os
import cv2
import numpy as np
from ultralytics import YOLO
import config

class PoseDetector:
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def track_frame(self, frame: np.ndarray):
        results = self.model.track(
            frame, 
            imgsz=config.IMGSZ, 
            conf=0.45, 
            iou=0.45, 
            classes=[0], 
            persist=True, 
            verbose=False
        )
        parsed = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            kps = results[0].keypoints.data.cpu().numpy()
            for tid, box, kp in zip(ids, boxes, kps):
                parsed.append((tid, tuple(box), kp))
        return parsed

class FaceRecognizer:
    def __init__(self, db_path: str = "face_database.pkl"):
        self.db = {}
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.db = pickle.load(f)

    def match_crop(self, face_crop: np.ndarray) -> str:
        if face_crop.size == 0 or not self.db:
            return "Unknown"
        # Dummy matching placeholder (simulating 512-d ArcFace vector comparison)
        return "Student"

class AsyncFaceWorker:
    def __init__(self, recognizer: FaceRecognizer):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.recognizer = recognizer

    def submit_recognition(self, track_id: int, face_crop: np.ndarray, callback_fn):
        def _task():
            name = self.recognizer.match_crop(face_crop)
            callback_fn(track_id, name)
        self.executor.submit(_task)