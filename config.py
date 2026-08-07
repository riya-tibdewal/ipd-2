"""Shared Constants & System Configuration"""
import numpy as np

# Performance & Inference
IMGSZ = 640  # Higher resolution significantly improves wrist & keypoint detection on desks
KP_CONF_THRESHOLD = 0.25  # Allows keypoint tracking even when partially occluded
STALE_TRACK_S = 7.0
DESIRED_FPS = 20

# Temporal Processing Window
TEMPORAL_WINDOW = 10

# Scale Fallback Ratio
BOX_WIDTH_SCALE_RATIO = 0.30

# Distance & Motion Thresholds
HEAD_DOWN_START = 0.75
HEAD_DOWN_STRONG = 0.15
HEAD_UP_START = 1.85
HEAD_UP_STRONG = 2.45
HEAD_SIDE_MIN = 0.25
HEAD_SIDE_MAX = 0.65
DESK_WRIST_REL_Y = 0.80
HAND_HEAD_DIST = 1.20
EXCESSIVE_MOTION = 0.28
WRITING_MIN = 0.015
WRITING_MAX = 0.22

# Temporal Hysteresis & EMA
EMA_ALPHA = 0.15
ATTENTION_SCORE_ALPHA = 0.15
MIN_SWITCH_FRAMES = 10
HYSTERESIS_MARGIN = 0.15
STRONG_EVIDENCE_FRAMES = 4
STRONG_EVIDENCE_MARGIN = 0.28
MIN_LABEL_CONFIDENCE = 0.42

# UI / Colors (BGR Format) - Primary Classes
BEHAVIOR_COLORS = {
    "WRITING": (0, 255, 0),        # Green (Attentive)
    "TALKING": (0, 165, 255),      # Amber/Orange (Non-Attentive)
    "USING_PHONE": (0, 0, 255),    # Red (Non-Attentive)
    "SLEEPING": (128, 0, 128),     # Purple (Non-Attentive)
    "UNKNOWN": (128, 128, 128)     # Grey
}

UPPER_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Facial landmarks
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left Arm
    (6, 8), (8, 10),                     # Right Arm
    (5, 11), (6, 12)                     # Torso
]