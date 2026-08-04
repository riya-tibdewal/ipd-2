"""4-Class Behavior Scoring Engine (Dev 2)"""
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

        if os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                self.tier1_model = bundle["tier1"]
                self.tier2_model = bundle["tier2"]
                self.label_map = bundle["label_map"]
                self.ml_enabled = True
                print(f"[BehaviorScoringEngine] Loaded 4-Class ML models from {model_path}")
            except Exception as e:
                print(f"[BehaviorScoringEngine] Failed to load ML models ({e}). Using heuristic fallback.")

    @staticmethod
    def _band_score(val: float, min_val: float, max_val: float) -> float:
        if min_val <= val <= max_val:
            return 1.0
        return 0.0

    def _metrics_to_feature_vector(self, m: PoseMetrics) -> np.ndarray:
        return np.array([[
            m.head_down, m.head_up, m.head_side, m.torso_lean,
            m.hands_near_desk, m.hands_near_head, m.wrist_motion, m.body_motion,
            m.vertical_extension, m.leg_visibility, m.back_view, m.seated_cue
        ]])

    def assess(self, m: PoseMetrics, prev_assessment: Optional[FrameAssessment] = None) -> FrameAssessment:
        assessment = FrameAssessment()

        wm = self._band_score(m.wrist_motion, config.WRITING_MIN, config.WRITING_MAX)
        prev_wm = prev_assessment.factors.get("rhythmic_wrist", 0.0) if prev_assessment else 0.0
        rhythmic_wrist = 0.45 * wm + 0.55 * prev_wm

        stillness = 1.0 - min(max(m.body_motion, m.wrist_motion) / config.EXCESSIVE_MOTION, 1.0)
        upright = (1.0 - m.torso_lean) * (1.0 - m.head_side)
        desk_focus = max(m.hands_near_desk, m.head_down * 0.65)

        assessment.factors = {
            "rhythmic_wrist": rhythmic_wrist,
            "stillness": stillness,
            "upright": upright,
            "desk_focus": desk_focus
        }

        # ML-Based 4-Class Evaluation
        if self.ml_enabled:
            feat_vec = self._metrics_to_feature_vector(m)
            probs = self.tier2_model.predict_proba(feat_vec)[0]
            classes = self.tier2_model.classes_
            
            scores: Dict[str, float] = {
                "WRITING": 0.0, "TALKING": 0.0, "USING_PHONE": 0.0,
                "SLEEPING": 0.0, "UNKNOWN": 0.0
            }

            for cls_idx, prob in zip(classes, probs):
                label_str = self.label_map.get(cls_idx, "UNKNOWN")
                scores[label_str] = float(prob)

            assessment.raw_scores = scores
            assessment.reasons = ["Trained 4-Class Random Forest Inference"]
            return assessment

        # Heuristic Fallback Engine
        scores = {
            "WRITING": rhythmic_wrist * 0.40 + desk_focus * 0.35 + m.head_down * 0.25,
            "USING_PHONE": m.hands_near_head * 0.50 + m.head_down * 0.30 + stillness * 0.20,
            "SLEEPING": m.head_down * 0.40 + stillness * 0.40 + (1.0 - upright) * 0.20,
            "TALKING": m.head_side * 0.60 + m.body_motion * 0.40,
            "UNKNOWN": 0.10
        }

        if scores["WRITING"] > 0.5 and desk_focus > 0.4:
            scores["SLEEPING"] *= 0.2
            scores["USING_PHONE"] *= 0.3

        assessment.raw_scores = scores
        assessment.reasons = [f"Primary cue active for {max(scores, key=scores.get)}"]
        return assessment