"""ByteTrack State, Hysteresis Buffer & 4-Class Attention Engine (Dev 3)"""
from collections import deque
import numpy as np
from typing import Dict, Tuple
from src.datatypes import FrameAssessment, ClassificationResult
from src.metrics import extract_pose_metrics
from src.scoring import BehaviorScoringEngine
import config

class StudentTemporalState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.history = deque(maxlen=60)
        self.ema_scores: Dict[str, float] = {}
        self.current_label: str = "WRITING"
        self.winning_candidate: str = "WRITING"
        self.candidate_lead_count: int = 0
        self.ema_attention: float = 85.0

    def update(self, assessment: FrameAssessment) -> Tuple[str, float, float]:
        self.history.append(assessment)
        raw = assessment.raw_scores

        for cls, score in raw.items():
            prev = self.ema_scores.get(cls, score)
            self.ema_scores[cls] = config.EMA_ALPHA * score + (1.0 - config.EMA_ALPHA) * prev

        top_cls = max(self.ema_scores, key=self.ema_scores.get)
        top_score = self.ema_scores[top_cls]
        curr_score = self.ema_scores.get(self.current_label, 0.0)
        margin = top_score - curr_score

        if top_cls != self.current_label:
            if top_cls == self.winning_candidate:
                self.candidate_lead_count += 1
            else:
                self.winning_candidate = top_cls
                self.candidate_lead_count = 1

            normal_switch = (self.candidate_lead_count >= config.MIN_SWITCH_FRAMES and margin >= config.HYSTERESIS_MARGIN)
            fast_switch = (self.candidate_lead_count >= config.STRONG_EVIDENCE_FRAMES and 
                           margin >= config.STRONG_EVIDENCE_MARGIN and 
                           top_score >= config.MIN_LABEL_CONFIDENCE)

            if normal_switch or fast_switch:
                self.current_label = top_cls
                self.candidate_lead_count = 0
        else:
            self.candidate_lead_count = 0

        # WRITING is Attentive, others are Non-Attentive
        f = assessment.factors
        writing_score = assessment.raw_scores.get("WRITING", 0.0)
        
        if self.current_label == "WRITING":
            raw_att = 85.0 + (writing_score * 15.0)
        elif self.current_label == "TALKING":
            raw_att = 35.0
        elif self.current_label in ["USING_PHONE", "SLEEPING"]:
            raw_att = 15.0
        else:
            raw_att = 50.0

        self.ema_attention = config.ATTENTION_SCORE_ALPHA * raw_att + (1.0 - config.ATTENTION_SCORE_ALPHA) * self.ema_attention
        return self.current_label, float(self.ema_scores[self.current_label]), float(self.ema_attention)

class TrackManager:
    def __init__(self):
        self.states: Dict[int, StudentTemporalState] = {}
        self.last_kps: Dict[int, np.ndarray] = {}
        self.scoring_engine = BehaviorScoringEngine()

    def process_student(
        self, 
        track_id: int, 
        bbox: Tuple[int, int, int, int], 
        kps: np.ndarray
    ) -> ClassificationResult:
        if track_id not in self.states:
            self.states[track_id] = StudentTemporalState(track_id)

        prev_kps = self.last_kps.get(track_id, None)
        metrics = extract_pose_metrics(kps, bbox, prev_kps)
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