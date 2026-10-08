# """Pose Keypoint Geometry Math (Dev 2) - High-Precision Extraction"""
# import numpy as np
# from typing import Tuple, Optional
# from src.datatypes import PoseMetrics
# import config

# def safe01(val: float) -> float:
#     return float(np.clip(val, 0.0, 1.0))

# def score_range(val: float, min_val: float, max_val: float) -> float:
#     if max_val == min_val:
#         return 0.0
#     return safe01((val - min_val) / (max_val - min_val))

# def extract_pose_metrics(
#     kps: np.ndarray, 
#     bbox: Tuple[int, int, int, int], 
#     prev_kps: Optional[np.ndarray] = None
# ) -> PoseMetrics:
#     metrics = PoseMetrics()
#     if kps is None or len(kps) < 17:
#         return metrics

#     x, y, conf = kps[:, 0], kps[:, 1], kps[:, 2]
#     x1, y1, x2, y2 = bbox
#     box_w = max(x2 - x1, 1.0)
#     box_h = max(y2 - y1, 1.0)

#     # Robust Scale Estimation using shoulders or hips
#     sh_w = np.linalg.norm([x[5] - x[6], y[5] - y[6]]) if (conf[5] > 0.2 and conf[6] > 0.2) else 0.0
#     hip_w = np.linalg.norm([x[11] - x[12], y[11] - y[12]]) if (conf[11] > 0.2 and conf[12] > 0.2) else 0.0
    
#     scale = sh_w if sh_w > 0 else (hip_w if hip_w > 0 else max(box_w * config.BOX_WIDTH_SCALE_RATIO, 1.0))
#     metrics.scale = float(scale)

#     # Face Keypoint Visibility (0 to 5)
#     face_vis_count = sum(1 for i in range(5) if conf[i] > 0.2)
#     metrics.back_view = 1.0 if (face_vis_count / 5.0) <= 0.20 else 0.0

#     # Advanced Head Geometry (Detect Sleeping/Collapsed Head)
#     if conf[0] > 0.2 and conf[5] > 0.2 and conf[6] > 0.2:
#         sh_mid_y = (y[5] + y[6]) / 2.0
#         # Distance from nose down to shoulder line relative to scale
#         nose_above_sh = (sh_mid_y - y[0]) / scale
#         metrics.head_down = score_range(nose_above_sh, config.HEAD_DOWN_START, config.HEAD_DOWN_STRONG)
#         metrics.head_up = score_range(nose_above_sh, config.HEAD_UP_START, config.HEAD_UP_STRONG)
#     elif face_vis_count <= 1 and conf[5] > 0.2 and conf[6] > 0.2:
#         # If facial keypoints disappear while sitting, head is collapsed/buried in arms
#         metrics.head_down = 0.90

#     # Head Side / Lateral Turning
#     if conf[0] > 0.2 and conf[3] > 0.2 and conf[4] > 0.2:
#         ear_mid_x = (x[3] + x[4]) / 2.0
#         metrics.head_side = score_range(abs(x[0] - ear_mid_x) / scale, config.HEAD_SIDE_MIN, config.HEAD_SIDE_MAX)
#     elif face_vis_count in [2, 3]:
#         metrics.head_side = 0.75

#     # Torso Lean
#     if conf[5] > 0.2 and conf[6] > 0.2 and conf[11] > 0.2 and conf[12] > 0.2:
#         sh_mid_x, sh_mid_y = (x[5] + x[6]) / 2.0, (y[5] + y[6]) / 2.0
#         hip_mid_x, hip_mid_y = (x[11] + x[12]) / 2.0, (y[11] + y[12]) / 2.0
#         metrics.torso_lean = safe01(abs(sh_mid_x - hip_mid_x) / max(hip_mid_y - sh_mid_y, 0.25) / scale)

#     # Hands Near Desk & Head
#     wrists_near_desk = 0
#     wrists_near_head = 0
#     valid_wrists = 0
    
#     for idx in [9, 10]:
#         if conf[idx] > 0.2:
#             valid_wrists += 1
#             rel_y = (y[idx] - y1) / box_h
#             if rel_y >= config.DESK_WRIST_REL_Y:
#                 wrists_near_desk += 1
#             if conf[0] > 0.2:
#                 dist_head = np.linalg.norm([x[idx] - x[0], y[idx] - y[0]]) / scale
#                 if dist_head <= config.HAND_HEAD_DIST:
#                     wrists_near_head += 1

#     if valid_wrists > 0:
#         metrics.hands_near_desk = wrists_near_desk / valid_wrists
#         metrics.hands_near_head = wrists_near_head / valid_wrists

#     # Temporal Wrist & Body Motion Calculation
#     if prev_kps is not None and len(prev_kps) == 17:
#         px, py, pconf = prev_kps[:, 0], prev_kps[:, 1], prev_kps[:, 2]
#         wrist_disps = []
#         for idx in [9, 10]:
#             if conf[idx] > 0.2 and pconf[idx] > 0.2:
#                 wrist_disps.append(np.linalg.norm([x[idx] - px[idx], y[idx] - py[idx]]))
#         if wrist_disps:
#             # Filter micro-jitter below 0.005
#             raw_speed = float(np.mean(wrist_disps) / scale)
#             metrics.wrist_motion = raw_speed if raw_speed > 0.005 else 0.0

#         body_idxs = [5, 6, 11, 12]
#         body_disps = [np.linalg.norm([x[i] - px[i], y[i] - py[i]]) for i in body_idxs if conf[i] > 0.2 and pconf[i] > 0.2]
#         if body_disps:
#             metrics.body_motion = float(np.mean(body_disps) / scale)

#     # Seated vs Standing Logic
#     if conf[5] > 0.2 and conf[6] > 0.2 and conf[11] > 0.2 and conf[12] > 0.2:
#         sh_mid_y = (y[5] + y[6]) / 2.0
#         hip_mid_y = (y[11] + y[12]) / 2.0
#         metrics.vertical_extension = safe01(((hip_mid_y - sh_mid_y) / box_h) * 2.2)

#     leg_kps = [13, 14, 15, 16]
#     metrics.leg_visibility = sum(1 for i in leg_kps if conf[i] > 0.2) / 4.0

#     if conf[11] > 0.2 and conf[12] > 0.2:
#         hip_low = max((y[11] - y1) / box_h, (y[12] - y1) / box_h)
#         metrics.seated_cue = safe01((hip_low - 0.58) / 0.22)

#     return metrics
"""Pose Keypoint Geometry Math - Standardized Feature Extraction"""
import numpy as np
from typing import Tuple, Optional
from src.datatypes import PoseMetrics
import config

def safe01(val: float) -> float:
    return float(np.clip(val, 0.0, 1.0))

def score_range(val: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.0
    return safe01((val - min_val) / (max_val - min_val))

def extract_pose_metrics(
    kps: np.ndarray, 
    bbox: Tuple[int, int, int, int], 
    prev_kps: Optional[np.ndarray] = None
) -> PoseMetrics:
    metrics = PoseMetrics()
    if kps is None or len(kps) < 17:
        return metrics

    x, y, conf = kps[:, 0], kps[:, 1], kps[:, 2]
    x1, y1, x2, y2 = bbox
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    # Scale calculation
    scale = max(box_h, 1.0)
    metrics.scale = float(scale)

    # Face visibility count
    face_vis_count = sum(1 for i in range(5) if conf[i] > 0.2)
    metrics.back_view = 1.0 if (face_vis_count / 5.0) <= 0.20 else 0.0

    # Head Down Gaze Geometry (Normalized by Bounding Box Height)
    if conf[0] > 0.2:
        nose_rel_y = (y[0] - y1) / box_h
        # Nose in top 40% of bounding box = Upright head
        # Nose below 60% of bounding box = Head down / sleeping
        metrics.head_down = score_range(nose_rel_y, 0.40, 0.70)
        metrics.head_up = score_range(0.40 - nose_rel_y, 0.0, 0.30)
    else:
        metrics.head_down = 0.0
        metrics.head_up = 0.0

    # Head Side / Facing Peer
    if conf[0] > 0.2 and conf[3] > 0.2 and conf[4] > 0.2:
        ear_mid_x = (x[3] + x[4]) / 2.0
        metrics.head_side = score_range(abs(x[0] - ear_mid_x) / box_w, 0.08, 0.30)
    elif face_vis_count in [2, 3]:
        metrics.head_side = 0.70

    # Torso Lean
    if conf[5] > 0.2 and conf[6] > 0.2 and conf[11] > 0.2 and conf[12] > 0.2:
        sh_mid_x, sh_mid_y = (x[5] + x[6]) / 2.0, (y[5] + y[6]) / 2.0
        hip_mid_x, hip_mid_y = (x[11] + x[12]) / 2.0, (y[11] + y[12]) / 2.0
        metrics.torso_lean = safe01(abs(sh_mid_x - hip_mid_x) / max(hip_mid_y - sh_mid_y, 0.25) / scale)

    # Hands Near Desk & Head
    wrists_near_desk = 0
    wrists_near_head = 0
    valid_wrists = 0
    
    for idx in [9, 10]:
        if conf[idx] > 0.2:
            valid_wrists += 1
            rel_y = (y[idx] - y1) / box_h
            if rel_y >= config.DESK_WRIST_REL_Y:
                wrists_near_desk += 1
            if conf[0] > 0.2:
                dist_head = np.linalg.norm([x[idx] - x[0], y[idx] - y[0]]) / box_h
                if dist_head <= 0.25:
                    wrists_near_head += 1

    if valid_wrists > 0:
        metrics.hands_near_desk = wrists_near_desk / valid_wrists
        metrics.hands_near_head = wrists_near_head / valid_wrists

    # Temporal Wrist & Body Motion
    if prev_kps is not None and len(prev_kps) == 17:
        px, py, pconf = prev_kps[:, 0], prev_kps[:, 1], prev_kps[:, 2]
        wrist_disps = []
        for idx in [9, 10]:
            if conf[idx] > 0.2 and pconf[idx] > 0.2:
                wrist_disps.append(np.linalg.norm([x[idx] - px[idx], y[idx] - py[idx]]))
        if wrist_disps:
            raw_speed = float(np.mean(wrist_disps) / box_h)
            metrics.wrist_motion = raw_speed if raw_speed > 0.005 else 0.0

        body_idxs = [5, 6, 11, 12]
        body_disps = [np.linalg.norm([x[i] - px[i], y[i] - py[i]]) for i in body_idxs if conf[i] > 0.2 and pconf[i] > 0.2]
        if body_disps:
            metrics.body_motion = float(np.mean(body_disps) / box_h)

    return metrics