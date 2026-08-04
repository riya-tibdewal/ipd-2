"""Dataclasses & Shared Data Structures (Dev 2)"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class PoseMetrics:
    head_down: float = 0.0
    head_up: float = 0.0
    head_side: float = 0.0
    torso_lean: float = 0.0
    hands_near_desk: float = 0.0
    hands_near_head: float = 0.0
    wrist_motion: float = 0.0
    body_motion: float = 0.0
    vertical_extension: float = 0.0
    leg_visibility: float = 0.0
    back_view: float = 0.0
    seated_cue: float = 0.0
    scale: float = 1.0

@dataclass
class FrameAssessment:
    raw_scores: Dict[str, float] = field(default_factory=dict)
    factors: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

@dataclass
class ClassificationResult:
    track_id: int
    name: str
    label: str
    confidence: float
    attention_score: float
    reasons: List[str]
    factors: Dict[str, float]
    raw_scores: Dict[str, float]
    bbox: Tuple[int, int, int, int]
    kps: np.ndarray