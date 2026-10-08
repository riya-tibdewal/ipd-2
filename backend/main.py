"""FastAPI service for uploading and analysing a classroom video."""
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.video_session import process_video

ROOT = Path(__file__).resolve().parents[1]
RUNTIME, UPLOADS, REPORTS = ROOT / "runtime", ROOT / "runtime" / "uploads", ROOT / "runtime" / "reports"
FRONTEND = ROOT / "frontend"
for directory in (UPLOADS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EduAware Analytics API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sessions: dict[str, dict[str, Any]] = {}
sessions_lock = Lock()
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def public_session(session: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in session.items() if key != "video_path"}


def run_session(session_id: str) -> None:
    with sessions_lock:
        session = sessions[session_id]
        session.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    def update_progress(done: int, total: int) -> None:
        with sessions_lock:
            sessions[session_id].update(processed_frames=done, total_frames=total,
                progress=round((done / total) * 100, 1) if total else 0.0)
    try:
        summary = process_video(session["video_path"], REPORTS / f"{session_id}.csv", update_progress)
        with sessions_lock:
            sessions[session_id].update(status="completed", progress=100.0, results=summary,
                report_url=f"/api/sessions/{session_id}/report",
                annotated_video_url=f"/api/sessions/{session_id}/video",
                completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        with sessions_lock:
            sessions[session_id].update(status="failed", error=str(error))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/sessions", status_code=201)
def create_session(video: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload an MP4, AVI, MOV, or MKV video.")
    session_id = uuid4().hex
    destination = UPLOADS / f"{session_id}{suffix}"
    with destination.open("wb") as target:
        shutil.copyfileobj(video.file, target)
    session = {"id": session_id, "filename": video.filename, "status": "queued", "progress": 0.0,
        "processed_frames": 0, "total_frames": 0, "results": [], "video_path": destination,
        "created_at": datetime.now(timezone.utc).isoformat()}
    with sessions_lock:
        sessions[session_id] = session
    return public_session(session)


@app.post("/api/sessions/{session_id}/start", status_code=202)
def start_session(session_id: str) -> dict[str, Any]:
    with sessions_lock:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found.")
        if session["status"] != "queued":
            raise HTTPException(409, f"Session is already {session['status']}.")
        Thread(target=run_session, args=(session_id,), daemon=True).start()
    return public_session(session)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    with sessions_lock:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found.")
        return public_session(session)


@app.get("/api/sessions/{session_id}/report")
def download_report(session_id: str):
    from fastapi.responses import FileResponse
    report = REPORTS / f"{session_id}.csv"
    if not report.exists():
        raise HTTPException(404, "Report is not ready.")
    return FileResponse(report, media_type="text/csv", filename="eduaaware-session-report.csv")


@app.get("/api/sessions/{session_id}/video")
def get_annotated_video(session_id: str):
    from fastapi.responses import FileResponse
    video = REPORTS / f"{session_id}_annotated.mp4"
    if not video.exists():
        raise HTTPException(404, "Annotated video is not ready.")
    return FileResponse(video, media_type="video/mp4")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

