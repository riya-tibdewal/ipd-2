# 
"""
Behavior Scoring Engine - Machine Learning Primary + Corrected Fallback
Uses trained Tier-1 / Tier-2 Random Forest models when available.
"""
import os
import joblib
import numpy as np
from typing import Optional, Dict
from src.datatypes import PoseMetrics, FrameAssessment
import config

class BehaviorScoringEngine:
    def __init__(self, model_path: str = "models/behavior_models.joblib"):
        self.ml_enabled = False
        self.tier1_model = None
        self.tier2_model = None
        self.label_map = {}

        # Load trained ML models if present
        if os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                self.tier1_model = bundle.get("tier1")
                self.tier2_model = bundle.get("tier2")
                raw_map = bundle.get("label_map", {})
                self.label_map = {int(k): v for k, v in raw_map.items()}
                
                if self.tier2_model is not None:
                    self.ml_enabled = True
                    print(f"[BehaviorScoringEngine] Successfully loaded ML pipeline from {model_path}")
            except Exception as e:
                print(f"[BehaviorScoringEngine] Failed to load ML models ({e}). Using heuristic fallback.")

    def _metrics_to_feature_vector(self, m: PoseMetrics) -> np.ndarray:
        """Converts PoseMetrics into the exact 14-feature vector required by trained models."""
        return np.array([[
            m.head_down, m.head_up, m.head_side, m.torso_lean,
            m.hands_near_desk, m.hands_near_head, m.wrist_motion, m.body_motion,
            m.vertical_extension, m.leg_visibility, m.back_view, m.seated_cue,
            getattr(m, 'has_phone', 0.0), getattr(m, 'has_notebook', 0.0)
        ]])

    def assess(self, m: PoseMetrics, prev_assessment: Optional[FrameAssessment] = None) -> FrameAssessment:
        assessment = FrameAssessment()

        has_phone = getattr(m, 'has_phone', 0.0) > 0.5
        has_notebook = getattr(m, 'has_notebook', 0.0) > 0.5
        wrist_speed = m.wrist_motion

        # -------------------------------------------------------------
        # PATH 1: TRAINED MACHINE LEARNING MODEL INFERENCE
        # -------------------------------------------------------------
        if self.ml_enabled and self.tier2_model is not None:
            feat_vec = self._metrics_to_feature_vector(m)
            import pandas as pd
            cols = [
                "head_down", "head_up", "head_side", "torso_lean",
                "hands_near_desk", "hands_near_head", "wrist_motion", "body_motion",
                "vertical_extension", "leg_visibility", "back_view", "seated_cue",
                "has_phone", "has_notebook"
            ]
            feat_df = pd.DataFrame(feat_vec, columns=cols)
            probs = self.tier2_model.predict_proba(feat_df)[0]
            classes = self.tier2_model.classes_


            scores: Dict[str, float] = {
                "WRITING": 0.0, "TALKING": 0.0, "USING_PHONE": 0.0,
                "SLEEPING": 0.0, "ATTENTIVE": 0.0
            }

            for cls_idx, prob in zip(classes, probs):
                label_str = self.label_map.get(int(cls_idx), "UNKNOWN")
                scores[label_str] = float(prob)

            # Strict Object Gating Overrides
            if has_phone:
                scores["USING_PHONE"] = max(scores["USING_PHONE"], 0.90)
                scores["WRITING"] *= 0.10
            elif not has_phone and scores["USING_PHONE"] > 0.70:
                # Suppress ghost phone predictions from ML if no phone object detected
                scores["USING_PHONE"] *= 0.20
                scores["TALKING"] = max(scores["TALKING"], 0.65)

            assessment.raw_scores = scores
            assessment.reasons = ["ML Random Forest Prediction"]
            return assessment

        # -------------------------------------------------------------
        # PATH 2: HEURISTIC FALLBACK (IF ML MODELS NOT FOUND)
        # -------------------------------------------------------------
        scores = {
            "WRITING": 0.0, "TALKING": 0.0, "USING_PHONE": 0.0,
            "SLEEPING": 0.0, "ATTENTIVE": 0.0
        }

        # Priority 1: True Sleeping (Nose collapsed deep in bounding box)
        if m.head_down > 0.80 and wrist_speed < 0.008:
            scores["SLEEPING"] = 0.95
            assessment.reasons = ["Collapsed Head + Zero Motion"]

        # Priority 2: Verified Phone Object
        elif has_phone and (m.hands_near_head > 0.35 or m.hands_near_desk > 0.4):
            scores["USING_PHONE"] = 0.95
            assessment.reasons = ["Confirmed Phone Object + Hand Position"]

        # Priority 3: Turning / Talking
        elif m.head_side > 0.45 or m.back_view > 0.40:
            scores["TALKING"] = 0.85
            assessment.reasons = ["Head/Body Turned Sideways"]

        # Priority 4: Active Writing
        elif (config.WRITING_MIN <= wrist_speed <= config.WRITING_MAX) and (m.hands_near_desk > 0.4 or has_notebook):
            scores["WRITING"] = 0.85
            assessment.reasons = ["Active Wrist Motion + Desk Focus"]

        # Priority 5: Default Upright Attentive
        else:
            scores["ATTENTIVE"] = 0.80
            assessment.reasons = ["Upright Forward Pose"]

        assessment.raw_scores = scores
        return assessment