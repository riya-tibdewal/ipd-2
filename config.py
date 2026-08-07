"""Shared Constants & System Configuration"""
import numpy as np

# Performance & Inference
IMGSZ = 640  # Higher resolution significantly improves wrist & keypoint detection on desks[cite: 10]
KP_CONF_THRESHOLD = 0.25 # Allows keypoint tracking even when partially occluded[cite: 10]
STALE_TRACK_S = 7.0[cite: 10]
DESIRED_FPS = 20[cite: 10]

# Temporal Processing Window
TEMPORAL_WINDOW = 10

# Scale Fallback Ratio
BOX_WIDTH_SCALE_RATIO = 0.30[cite: 10]

# Distance & Motion Thresholds
HEAD_DOWN_START = 0.75[cite: 10]
HEAD_DOWN_STRONG = 0.15[cite: 10]
HEAD_UP_START = 1.85[cite: 10]
HEAD_UP_STRONG = 2.45[cite: 10]
HEAD_SIDE_MIN = 0.25[cite: 10]
HEAD_SIDE_MAX = 0.65[cite: 10]
DESK_WRIST_REL_Y = 0.80[cite: 10]
HAND_HEAD_DIST = 1.20[cite: 10]
EXCESSIVE_MOTION = 0.28[cite: 10]
WRITING_MIN = 0.015[cite: 10]
WRITING_MAX = 0.22[cite: 10]

# Temporal Hysteresis & EMA
EMA_ALPHA = 0.15[cite: 10]
ATTENTION_SCORE_ALPHA = 0.15[cite: 10]
MIN_SWITCH_FRAMES = 10[cite: 10]
HYSTERESIS_MARGIN = 0.15[cite: 10]
STRONG_EVIDENCE_FRAMES = 4[cite: 10]
STRONG_EVIDENCE_MARGIN = 0.28[cite: 10]
MIN_LABEL_CONFIDENCE = 0.42[cite: 10]

# UI / Colors (BGR Format) - Primary Classes
BEHAVIOR_COLORS = {
    "WRITING": (0, 255, 0),        # Green (Attentive)[cite: 10]
    "TALKING": (0, 165, 255),      # Amber/Orange (Non-Attentive)[cite: 10]
    "USING_PHONE": (0, 0, 255),    # Red (Non-Attentive)[cite: 10]
    "SLEEPING": (128, 0, 128),     # Purple (Non-Attentive)[cite: 10]
    "UNKNOWN": (128, 128, 128)     # Grey[cite: 10]
}

UPPER_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Facial landmarks[cite: 10]
    (5, 6),                              # Shoulders[cite: 10]
    (5, 7), (7, 9),                      # Left Arm[cite: 10]
    (6, 8), (8, 10),                     # Right Arm[cite: 10]
    (5, 11), (6, 12)                     # Torso[cite: 10]
]