"""YOLOv8 Pose Engine (Dev 1 - Pose Tracking Only)"""
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