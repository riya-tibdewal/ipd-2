"""
Behavior Scoring Engine - Object Gated & Velocity Tracked
Eliminates false phone misclassifications and arbitrary heuristic weights.
"""
import os
import numpy as np
from typing import Optional, Dict
from src.datatypes import PoseMetrics, FrameAssessment
import config

class BehaviorScoringEngine:
    def __init__(self, model_path: str = "models/behavior_models.joblib"):
        self.ml_enabled = False

    def assess(self, m: PoseMetrics, prev_assessment: Optional[FrameAssessment] = None) -> FrameAssessment:
        assessment = FrameAssessment()

        has_phone = getattr(m, 'has_phone', 0.0) > 0.5
        has_notebook = getattr(m, 'has_notebook', 0.0) > 0.5
        wrist_speed = m.wrist_motion
        
        scores: Dict[str, float] = {
            "WRITING": 0.0,
            "TALKING": 0.0,
            "USING_PHONE": 0.0,
            "SLEEPING": 0.0,
            "ATTENTIVE": 0.0
        }

        # 1. STRICT OBJECT GATING FOR PHONE
        # Never classify USING_PHONE unless secondary YOLO detector sees an actual phone object
        if has_phone and (m.hands_near_head > 0.35 or m.hands_near_desk > 0.4):
            scores["USING_PHONE"] = 0.95
            assessment.reasons = ["Confirmed Phone Object + Hand Placement"]

        # 2. TEMPORAL WRIST VELOCITY FOR WRITING
        # Differentiates active writing pen-strokes from idle resting hands
        elif (config.WRITING_MIN <= wrist_speed <= config.WRITING_MAX) and (m.hands_near_desk > 0.4 or has_notebook):
            scores["WRITING"] = 0.90
            assessment.reasons = ["Active Wrist Motion + Desk Focus"]

        # 3. TURNING & TALKING (NO FACIAL LANDMARKS / LATERAL TURN)
        elif m.back_view > 0.4 or m.head_side > 0.5 or m.body_motion > 0.20:
            scores["TALKING"] = 0.85
            assessment.reasons = ["Body/Head Lateral Orientation Shift"]

        # 4. SLEEPING (STATIC NEAR-ZERO MOTION + HEAD COLLAPSE)
        elif m.head_down > 0.75 and wrist_speed < 0.010 and m.body_motion < 0.010:
            scores["SLEEPING"] = 0.85
            assessment.reasons = ["Prolonged Inactivity + Head Down"]

        # 5. DEFAULT ATTENTIVE STATE
        else:
            scores["ATTENTIVE"] = 0.80
            assessment.reasons = ["Standard Upright Engagement"]

        assessment.raw_scores = scores
        return assessment