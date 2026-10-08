"""Feature extraction used by the web pipeline and optimized trainer."""
from typing import Optional, Tuple

import numpy as np

from src.datatypes import PoseMetrics
from src.metrics import extract_pose_metrics, safe01


FEATURE_NAMES = (
    "head_down", "head_up", "head_side", "torso_lean",
    "hands_near_desk", "hands_near_head", "wrist_motion", "body_motion",
    "vertical_extension", "leg_visibility", "back_view", "seated_cue",
    "has_phone", "has_notebook",
)


def extract_optimized_metrics(
    kps: np.ndarray,
    bbox: Tuple[int, int, int, int],
    prev_kps: Optional[np.ndarray] = None,
) -> PoseMetrics:
    """Return the complete, stable feature set for one detected person."""
    metrics = extract_pose_metrics(kps, bbox, prev_kps)
    if kps is None or len(kps) < 17:
        return metrics

    x, y, conf = kps[:, 0], kps[:, 1], kps[:, 2]
    x1, y1, x2, y2 = bbox
    box_h = max(y2 - y1, 1.0)

    # Correct the double-normalisation in the legacy torso calculation.
    if all(conf[index] > 0.2 for index in (5, 6, 11, 12)):
        shoulder_x = (x[5] + x[6]) / 2.0
        shoulder_y = (y[5] + y[6]) / 2.0
        hip_x = (x[11] + x[12]) / 2.0
        hip_y = (y[11] + y[12]) / 2.0
        torso_height = max(hip_y - shoulder_y, 1.0)
        metrics.torso_lean = safe01(abs(shoulder_x - hip_x) / torso_height)
        metrics.vertical_extension = safe01(((hip_y - shoulder_y) / box_h) * 2.2)

    metrics.leg_visibility = sum(conf[index] > 0.2 for index in (13, 14, 15, 16)) / 4.0
    if conf[11] > 0.2 and conf[12] > 0.2:
        hip_low = max((y[11] - y1) / box_h, (y[12] - y1) / box_h)
        metrics.seated_cue = safe01((hip_low - 0.58) / 0.22)
    return metrics


def feature_vector(metrics: PoseMetrics) -> np.ndarray:
    """Create the feature vector shared by optimized training and inference."""
    return np.asarray([[getattr(metrics, name) for name in FEATURE_NAMES]], dtype=float)
