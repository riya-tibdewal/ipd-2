# # 

# """Student Tracking & Temporal State Manager"""
# from typing import Dict, List, Tuple, Optional
# import numpy as np

# from src.datatypes import ClassificationResult, FrameAssessment, PoseMetrics
# from src.metrics import extract_pose_metrics
# from src.scoring import BehaviorScoringEngine
# import config


# class StudentTemporalState:
#     """Manages 10-frame hysteresis window for a single student track."""
#     def __init__(self, track_id: int):
#         self.track_id = track_id
#         self.history: List[FrameAssessment] = []
#         self.max_window = config.TEMPORAL_WINDOW

#     def update(self, assessment: FrameAssessment) -> Tuple[str, float, float]:
#         self.history.append(assessment)
#         if len(self.history) > self.max_window:
#             self.history.pop(0)

#         # Average probabilities across temporal window
#         class_sums: Dict[str, float] = {}
#         for item in self.history:
#             for cls_name, score in item.raw_scores.items():
#                 class_sums[cls_name] = class_sums.get(cls_name, 0.0) + score

#         n_frames = len(self.history)
#         avg_scores = {k: v / n_frames for k, v in class_sums.items()}

#         top_label = max(avg_scores, key=avg_scores.get)
#         confidence = avg_scores[top_label]

#         # Calculate scaled attention score
#         if top_label == "WRITING":
#             att_score = 80.0 + confidence * 20.0
#         elif top_label == "TALKING":
#             att_score = 30.0 + (1.0 - confidence) * 20.0
#         elif top_label == "USING_PHONE":
#             att_score = 10.0 + (1.0 - confidence) * 15.0
#         elif top_label == "SLEEPING":
#             att_score = 5.0 + (1.0 - confidence) * 10.0
#         else:
#             att_score = 50.0

#         return top_label, confidence, min(max(att_score, 0.0), 100.0)


# class TrackManager:
#     """Coordinates detection, temporal smoothing, and classification across tracks."""
#     def __init__(self):
#         self.states: Dict[int, StudentTemporalState] = {}
#         self.last_kps: Dict[int, np.ndarray] = {}
#         self.scoring_engine = BehaviorScoringEngine()

#     def process_student(
#         self, 
#         track_id: int, 
#         bbox: Tuple[int, int, int, int], 
#         kps: np.ndarray,
#         has_phone: bool = False,
#         has_notebook: bool = False
#     ) -> ClassificationResult:
#         if track_id not in self.states:
#             self.states[track_id] = StudentTemporalState(track_id)

#         prev_kps = self.last_kps.get(track_id, None)
#         metrics = extract_pose_metrics(kps, bbox, prev_kps)
        
#         # Attach object detection flags to metrics
#         metrics.has_phone = 1.0 if has_phone else 0.0
#         metrics.has_notebook = 1.0 if has_notebook else 0.0

#         self.last_kps[track_id] = kps.copy()

#         prev_assessment = self.states[track_id].history[-1] if self.states[track_id].history else None
#         assessment = self.scoring_engine.assess(metrics, prev_assessment)
        
#         label, conf, att_score = self.states[track_id].update(assessment)
#         name = f"Student-{track_id}"

#         return ClassificationResult(
#             track_id=track_id,
#             name=name,
#             label=label,
#             confidence=conf,
#             attention_score=att_score,
#             reasons=assessment.reasons,
#             factors=assessment.factors,
#             raw_scores=assessment.raw_scores,
#             bbox=bbox,
#             kps=kps
#         )
"""Student Tracking & Temporal State Manager"""
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.datatypes import ClassificationResult, FrameAssessment, PoseMetrics
from src.metrics import extract_pose_metrics
from src.scoring import BehaviorScoringEngine
import config

class StudentTemporalState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.history: List[FrameAssessment] = []
        self.max_window = config.TEMPORAL_WINDOW

    def update(self, assessment: FrameAssessment) -> Tuple[str, float, float]:
        self.history.append(assessment)
        if len(self.history) > self.max_window:
            self.history.pop(0)

        class_sums: Dict[str, float] = {}
        for item in self.history:
            for cls_name, score in item.raw_scores.items():
                class_sums[cls_name] = class_sums.get(cls_name, 0.0) + score

        n_frames = len(self.history)
        avg_scores = {k: v / n_frames for k, v in class_sums.items()}

        top_label = max(avg_scores, key=avg_scores.get)
        confidence = avg_scores[top_label]

        if top_label == "WRITING":
            att_score = 80.0 + confidence * 20.0
        elif top_label == "TALKING":
            att_score = 30.0 + (1.0 - confidence) * 20.0
        elif top_label == "USING_PHONE":
            att_score = 10.0 + (1.0 - confidence) * 15.0
        elif top_label == "SLEEPING":
            att_score = 5.0 + (1.0 - confidence) * 10.0
        else:
            att_score = 50.0

        return top_label, confidence, min(max(att_score, 0.0), 100.0)

class TrackManager:
    def __init__(self):
        self.states: Dict[int, StudentTemporalState] = {}
        self.last_kps: Dict[int, np.ndarray] = {}
        self.scoring_engine = BehaviorScoringEngine()

    def process_student(
        self, 
        track_id: int, 
        bbox: Tuple[int, int, int, int], 
        kps: np.ndarray,
        has_phone: bool = False,
        has_notebook: bool = False
    ) -> ClassificationResult:
        if track_id not in self.states:
            self.states[track_id] = StudentTemporalState(track_id)

        prev_kps = self.last_kps.get(track_id, None)
        metrics = extract_pose_metrics(kps, bbox, prev_kps)
        
        metrics.has_phone = 1.0 if has_phone else 0.0
        metrics.has_notebook = 1.0 if has_notebook else 0.0

        self.last_kps[track_id] = kps.copy()

        prev_assessment = self.states[track_id].history[-1] if self.states[track_id].history else None
        assessment = self.scoring_engine.assess(metrics, prev_assessment)
        
        label, conf, att_score = self.states[track_id].update(assessment)
        name = f"Student-{track_id}"

        return ClassificationResult(
            track_id=track_id,
            name=name,
            label=label,
            confidence=conf,
            attention_score=att_score,
            reasons=assessment.reasons,
            factors=assessment.factors,
            raw_scores=assessment.raw_scores,
            bbox=bbox,
            kps=kps
        )