"""Time-based tracking state for the web/demo pipeline."""
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from src.datatypes import ClassificationResult, FrameAssessment
from src.optimized_metrics import extract_optimized_metrics
from src.scoring import BehaviorScoringEngine


@dataclass
class OptimizedStudentState:
    track_id: int
    history: Deque[Tuple[float, FrameAssessment]] = field(default_factory=deque)
    last_seen: float = 0.0
    last_kps: Optional[np.ndarray] = None
    current_label: str = "UNKNOWN"
    current_confidence: float = 0.0
    attention_total: float = 0.0
    observation_count: int = 0
    label_counts: Counter = field(default_factory=Counter)


class OptimizedTrackManager:
    """Smooth predictions over elapsed time and retain reportable session state."""
    def __init__(self, window_seconds: float = 2.0, stale_track_seconds: float = 7.0):
        self.window_seconds = window_seconds
        self.stale_track_seconds = stale_track_seconds
        self.states: Dict[int, OptimizedStudentState] = {}
        root = Path(__file__).resolve().parents[1]
        optimized_model = root / "models" / "optimized_behavior_models.joblib"
        legacy_model = root / "models" / "behavior_models.joblib"
        self.scoring_engine = BehaviorScoringEngine(str(optimized_model if optimized_model.exists() else legacy_model))

    @staticmethod
    def _attention_score(label: str, confidence: float) -> float:
        base_scores = {"WRITING": 90.0, "ATTENTIVE": 80.0, "TALKING": 35.0,
                       "USING_PHONE": 15.0, "SLEEPING": 5.0}
        return max(0.0, min(100.0, base_scores.get(label, 50.0) + (confidence - 0.5) * 10.0))

    def _expire_stale_histories(self, now: float) -> None:
        for state in self.states.values():
            if now - state.last_seen > self.stale_track_seconds:
                state.history.clear()
                state.last_kps = None

    def process_student(self, track_id: int, bbox: tuple, kps: np.ndarray,
                        has_phone: bool = False, has_notebook: bool = False,
                        timestamp: Optional[float] = None) -> ClassificationResult:
        now = monotonic() if timestamp is None else timestamp
        self._expire_stale_histories(now)
        state = self.states.setdefault(track_id, OptimizedStudentState(track_id))
        metrics = extract_optimized_metrics(kps, bbox, state.last_kps)
        metrics.has_phone = float(has_phone)
        metrics.has_notebook = float(has_notebook)
        state.last_kps, state.last_seen = kps.copy(), now

        assessment = self.scoring_engine.assess(metrics)
        state.history.append((now, assessment))
        while state.history and now - state.history[0][0] > self.window_seconds:
            state.history.popleft()

        scores = {label: 0.0 for label in ("WRITING", "TALKING", "USING_PHONE", "SLEEPING", "ATTENTIVE")}
        for _, item in state.history:
            for label, score in item.raw_scores.items():
                scores[label] = scores.get(label, 0.0) + score
        count = max(len(state.history), 1)
        scores = {label: score / count for label, score in scores.items()}
        label, confidence = max(scores.items(), key=lambda pair: pair[1])
        attention = self._attention_score(label, confidence)

        state.current_label, state.current_confidence = label, confidence
        state.attention_total += attention
        state.observation_count += 1
        state.label_counts[label] += 1
        return ClassificationResult(track_id, f"Student-{track_id}", label, confidence,
                                    attention, assessment.reasons, assessment.factors,
                                    scores, bbox, kps)

    def session_summary(self) -> list[dict]:
        return [{
            "track_id": track_id,
            "name": f"Student-{track_id}",
            "final_label": state.current_label,
            "final_confidence": round(state.current_confidence, 3),
            "average_attention": round(state.attention_total / max(state.observation_count, 1), 2),
            "observations": state.observation_count,
            "label_counts": dict(state.label_counts),
        } for track_id, state in sorted(self.states.items())]
