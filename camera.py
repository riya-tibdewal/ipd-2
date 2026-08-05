# """
# Dataset Generator & 4-Class Behavior Classifier Trainer
# Trains Tier-1 (WRITING vs REST) and Tier-2 (4 Classes: WRITING, TALKING, USING_PHONE, SLEEPING).
# """
# import os
# import cv2
# import glob
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# from src.ai_engine import PoseDetector
# from src.metrics import extract_pose_metrics

# # Strict 4-Class Taxonomy Mapping
# LABEL_MAP = {
#     "WRITING": 0,      # Attentive (Target Tier-1 = 0)
#     "TALKING": 1,      # Non-Attentive (Target Tier-1 = 1)
#     "USING_PHONE": 2,  # Non-Attentive (Target Tier-1 = 1)
#     "SLEEPING": 3      # Non-Attentive (Target Tier-1 = 1)
# }
# INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# def extract_features_from_videos(video_dir="data/training_videos", output_csv="data/pose_dataset.csv"):
#     detector = PoseDetector("yolov8n-pose.pt")
#     data_rows = []
    
#     video_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True) + \
#                   glob.glob(os.path.join(video_dir, "**", "*.avi"), recursive=True)

#     if not video_files:
#         print(f"No video files found in {video_dir}! Structure as: {video_dir}/CLASS_NAME/video.mp4")
#         return None

#     print(f"Found {len(video_files)} video clips across 4 behavior classes. Extracting features...")

#     for video_path in video_files:
#         label_str = os.path.basename(os.path.dirname(video_path)).upper()
#         if label_str not in LABEL_MAP:
#             continue
        
#         cap = cv2.VideoCapture(video_path)
#         prev_kps_dict = {}

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             detections = detector.track_frame(frame)
#             for tid, bbox, kps in detections:
#                 prev_kps = prev_kps_dict.get(tid, None)
#                 metrics = extract_pose_metrics(kps, bbox, prev_kps)
#                 prev_kps_dict[tid] = kps.copy()

#                 feat = [
#                     metrics.head_down, metrics.head_up, metrics.head_side, metrics.torso_lean,
#                     metrics.hands_near_desk, metrics.hands_near_head, metrics.wrist_motion, metrics.body_motion,
#                     metrics.vertical_extension, metrics.leg_visibility, metrics.back_view, metrics.seated_cue,
#                     LABEL_MAP[label_str]
#                 ]
#                 data_rows.append(feat)

#         cap.release()

#     cols = [
#         "head_down", "head_up", "head_side", "torso_lean",
#         "hands_near_desk", "hands_near_head", "wrist_motion", "body_motion",
#         "vertical_extension", "leg_visibility", "back_view", "seated_cue",
#         "target_label"
#     ]
#     df = pd.DataFrame(data_rows, columns=cols)
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     df.to_csv(output_csv, index=False)
#     print(f"Saved {len(df)} feature frames to {output_csv}")
#     return df

# def train_models(csv_path="data/pose_dataset.csv", model_out="models/behavior_models.joblib"):
#     if not os.path.exists(csv_path):
#         print("Dataset CSV missing! Extract features first.")
#         return

#     df = pd.read_csv(csv_path)
#     X = df.drop(columns=["target_label"])
#     y = df["target_label"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     # 1. Tier 1 Binary Classifier (0: Attentive [WRITING], 1: Non-Attentive [REST])
#     y_train_t1 = (y_train != 0).astype(int)
#     y_test_t1 = (y_test != 0).astype(int)

#     tier1_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
#     tier1_model.fit(X_train, y_train_t1)
#     print("\n--- Tier-1 Classifier Performance (Attentive [Writing] vs Non-Attentive) ---")
#     print(classification_report(y_test_t1, tier1_model.predict(X_test), target_names=["Attentive (Writing)", "Non-Attentive"]))

#     # 2. Tier 2 Multi-Class Classifier (4 Classes)
#     tier2_model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
#     tier2_model.fit(X_train, y_train)
#     print("\n--- Tier-2 Multi-Class Classifier Performance (4 Classes) ---")
#     print(classification_report(y_test, tier2_model.predict(X_test), target_names=[INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP.keys())]))

#     os.makedirs(os.path.dirname(model_out), exist_ok=True)
#     joblib.dump({
#         "tier1": tier1_model,
#         "tier2": tier2_model,
#         "label_map": INV_LABEL_MAP
#     }, model_out)
#     print(f"Trained 4-class pipeline saved to {model_out}")

# if __name__ == "__main__":
#     extract_features_from_videos()
#     train_models()

"""Video Feed & Capture Manager"""
import cv2

class CameraManager:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame

    def release(self):
        self.cap.release()