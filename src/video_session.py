"""Non-UI video processing service shared by the API and future clients."""
from pathlib import Path
from time import monotonic
from typing import Callable, Optional

import cv2

from src.ai_engine import PoseDetector
from src.optimized_tracker import OptimizedTrackManager
from src.session_reporting import export_session_csv
from src.visualization import Visualizer

ProgressCallback = Callable[[int, int], None]


def process_video(video_path: Path, report_path: Path, progress: Optional[ProgressCallback] = None) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    annotated_video_path = report_path.with_name(f"{report_path.stem}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    detector = PoseDetector("yolov8n-pose.pt", imgsz=384)
    tracker = OptimizedTrackManager()
    vis = Visualizer()
    frame_number = 0
    prev_time = monotonic()
    last_frame_results = []
    frame_stride = 2

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else monotonic()

            curr_time = monotonic()
            calc_fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            if frame_number == 1 or frame_number % frame_stride == 0:
                detections = detector.track_frame(frame)
                last_frame_results = []
                for track_id, bbox, kps, has_phone, has_notebook in detections:
                    res = tracker.process_student(track_id, bbox, kps, has_phone, has_notebook, timestamp)
                    last_frame_results.append(res)
                    vis.draw_student(frame, res)
            else:
                for res in last_frame_results:
                    vis.draw_student(frame, res)

            vis.draw_hud(frame, calc_fps, last_frame_results)
            if video_writer.isOpened():
                video_writer.write(frame)

            if progress and (frame_number == 1 or frame_number % 10 == 0):
                progress(frame_number, total_frames)
    finally:
        capture.release()
        if video_writer:
            video_writer.release()

    export_session_csv(report_path, tracker)
    if progress:
        progress(frame_number, total_frames)
    return tracker.session_summary()


