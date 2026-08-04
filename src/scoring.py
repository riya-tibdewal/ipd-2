"""Tier 1 & Tier 2 Classifiers / Heuristics Engine (Dev 2)"""
import numpy as np
from src.datatypes import PoseMetrics, FrameAssessment
import config

class BehaviorScoringEngine:
    @staticmethod
    def _band_score(val: float, min_val: float, max_val: float) -> float:
        if min_val <= val <= max_val:
            return 1.0
        return 0.0

    def assess(self, m: PoseMetrics, prev_assessment: Optional[FrameAssessment] = None) -> FrameAssessment:
        assessment = FrameAssessment()
        
        # Intermediate Factors
        wm = self._band_score(m.wrist_motion, config.WRITING_MIN, config.WRITING_MAX)
        prev_wm = prev_assessment.factors.get("rhythmic_wrist", 0.0) if prev_assessment else 0.0
        rhythmic_wrist = 0.45 * wm + 0.55 * prev_wm
        
        stillness = 1.0 - min(max(m.body_motion, m.wrist_motion) / config.EXCESSIVE_MOTION, 1.0)
        upright = (1.0 - m.torso_lean) * (1.0 - m.head_side)
        desk_focus = max(m.hands_near_desk, m.head_down * 0.65)
        standing_evidence = max(m.vertical_extension, m.leg_visibility) * upright * (1.0 - m.seated_cue)

        factors = {
            "rhythmic_wrist": rhythmic_wrist,
            "stillness": stillness,
            "upright": upright,
            "desk_focus": desk_focus,
            "standing_evidence": standing_evidence
        }
        assessment.factors = factors

        # Base Scores for 8 Classes
        scores = {
            "ATTENTIVE": upright * (1.0 - m.head_down) * (1.0 - m.head_side) * 0.85,
            "WRITING": rhythmic_wrist * 0.40 + desk_focus * 0.35 + m.head_down * 0.25,
            "USING_PHONE": m.hands_near_head * 0.50 + m.head_down * 0.30 + stillness * 0.20,
            "SLEEPING": m.head_down * 0.40 + stillness * 0.40 + (1.0 - upright) * 0.20,
            "STANDING": standing_evidence,
            "TALKING": m.head_side * 0.60 + m.body_motion * 0.40,
            "DISTRACTED": max(m.head_side, m.head_up) * 0.70 + (1.0 - stillness) * 0.30,
            "UNKNOWN": 0.10
        }

        # Competition & Mutual Suppression Rules
        if scores["WRITING"] > 0.5 and desk_focus > 0.4:
            scores["SLEEPING"] *= 0.2
            scores["USING_PHONE"] *= 0.3

        if scores["SLEEPING"] > 0.6 and stillness > 0.7:
            scores["WRITING"] *= 0.1
            scores["STANDING"] *= 0.0

        if not any(v > 0.52 for k, v in scores.items() if k != "ATTENTIVE"):
            scores["ATTENTIVE"] = max(scores["ATTENTIVE"], 0.68)

        assessment.raw_scores = scores
        assessment.reasons = [f"Primary cue active for {max(scores, key=scores.get)}"]
        return assessment