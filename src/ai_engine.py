"""YOLOv8 Pose + Object Engine (Phone & Notebook Detection)"""
import os
import numpy as np
from ultralytics import YOLO
import config

class PoseDetector:
    def __init__(self, pose_model_path: str = "yolov8s-pose.pt", obj_model_path: str = "yolov8n.pt"):
        # Primary Pose Model
        if os.path.exists(pose_model_path):
            self.pose_model = YOLO(pose_model_path)
        else:
            self.pose_model = YOLO("yolov8n-pose.pt")

        # Secondary Object Model (Cell phones & Books/Notebooks)
        if obj_model_path and os.path.exists(obj_model_path) and os.path.getsize(obj_model_path) > 1000000:
            self.obj_model = YOLO(obj_model_path)
        else:
            self.obj_model = None

    def track_frame(self, frame: np.ndarray):
        results = self.pose_model.track(
            frame, 
            imgsz=config.IMGSZ, 
            conf=0.45, 
            iou=0.45, 
            classes=[0], # Person class
            persist=True, 
            verbose=False
        )
        
        parsed = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            kps = results[0].keypoints.data.cpu().numpy()
            
            phone_boxes = []
            notebook_boxes = []

            if self.obj_model is not None:
                # Class 67: Cell Phone | Class 73: Book/Notebook
                obj_results = self.obj_model(frame, imgsz=config.IMGSZ, conf=0.25, classes=[67, 73], verbose=False)
                if obj_results and obj_results[0].boxes is not None:
                    obj_boxes = obj_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    obj_classes = obj_results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for obox, ocls in zip(obj_boxes, obj_classes):
                        if ocls == 67:
                            phone_boxes.append(obox)
                        elif ocls == 73:
                            notebook_boxes.append(obox)

            for tid, box, kp in zip(ids, boxes, kps):
                x1, y1, x2, y2 = box
                
                # Check Phone Bounding Box Overlap
                has_phone = False
                for px1, py1, px2, py2 in phone_boxes:
                    if not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2):
                        has_phone = True
                        break

                # Check Notebook Bounding Box Overlap
                has_notebook = False
                for nx1, ny1, nx2, ny2 in notebook_boxes:
                    if not (nx2 < x1 or nx1 > x2 or ny2 < y1 or ny1 > y2):
                        has_notebook = True
                        break

                parsed.append((tid, tuple(box), kp, has_phone, has_notebook))
        return parsed