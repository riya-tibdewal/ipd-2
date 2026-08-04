"""Skeleton, Bounding Boxes & Live HUD Overlay (Dev 4)"""
import cv2
import numpy as np
from typing import List, Dict
from src.datatypes import ClassificationResult
import config

class Visualizer:
    def __init__(self):
        self.bbox_histories: Dict[int, List[np.ndarray]] = {}

    def smooth_bbox(self, track_id: int, bbox: tuple) -> tuple:
        if track_id not in self.bbox_histories:
            self.bbox_histories[track_id] = []
        hist = self.bbox_histories[track_id]
        hist.append(np.array(bbox))
        if len(hist) > 12:
            hist.pop(0)
        
        weights = np.arange(1, len(hist) + 1)
        smoothed = np.average(hist, axis=0, weights=weights).astype(int)
        return tuple(smoothed)

    def draw_student(self, frame: np.ndarray, res: ClassificationResult):
        x1, y1, x2, y2 = self.smooth_bbox(res.track_id, res.bbox)
        color = config.BEHAVIOR_COLORS.get(res.label, (128, 128, 128))

        # Box Overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Upper-Body Skeleton
        kps = res.kps
        for p1, p2 in config.UPPER_SKELETON_EDGES:
            if kps[p1][2] >= config.KP_CONF_THRESHOLD and kps[p2][2] >= config.KP_CONF_THRESHOLD:
                pt1 = (int(kps[p1][0]), int(kps[p1][1]))
                pt2 = (int(kps[p2][0]), int(kps[p2][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

        # HUD Label Pill
        label_text = f"{res.name} | {res.label} | Attn: {res.attention_score:.1f}%"
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_hud(self, frame: np.ndarray, fps: float, results: List[ClassificationResult]):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        
        avg_attn = np.mean([r.attention_score for r in results]) if results else 0.0
        hud_text = f"FPS: {fps:.1f} | Active Students: {len(results)} | Class Avg Attn: {avg_attn:.1f}%"
        
        status_color = (0, 255, 0) if avg_attn >= 70 else ((0, 165, 255) if avg_attn >= 45 else (0, 0, 255))
        cv2.putText(frame, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)