"""Production-oriented API and background worker for Shorts Maker.

The API only creates and reports jobs. CPU/GPU-heavy processing runs in a
bounded executor, keeping FastAPI responsive when several users submit work.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from urllib.error import URLError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

load_dotenv()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


APP_NAME = "shorts-generator"
DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
SOURCES_DIR = DATA_DIR / "sources"
OUTPUTS_DIR = DATA_DIR / "outputs"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
TEMP_DIR = DATA_DIR / "temp"
JOBS_FILE = DATA_DIR / "jobs.json"
MAX_WORKERS = max(1, env_int("MAX_CONCURRENT_JOBS", 1))
MAX_UPLOAD_MB = max(1, env_int("MAX_UPLOAD_MB", 2048))
MAX_CLIPS = max(1, min(8, env_int("MAX_CLIPS", 3)))
OUTPUT_WIDTH = max(360, env_int("OUTPUT_WIDTH", 1080))
OUTPUT_HEIGHT = max(640, env_int("OUTPUT_HEIGHT", 1920))
VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "libx264")
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "medium")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CORS_ORIGINS = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",")]

for directory in (DATA_DIR, UPLOADS_DIR, SOURCES_DIR, OUTPUTS_DIR, THUMBNAILS_DIR, TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Shorts Generator API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=CORS_ORIGINS != ["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.RLock()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="shorts-worker")
whisper_model: WhisperModel | None = None
whisper_lock = threading.Lock()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CAPTION_STYLES = {
    "hormozi": {"primary": "&H0000FFFF&", "outline": 4, "size": 68},
    "neon": {"primary": "&H00FFFF00&", "outline": 4, "size": 66},
    "comic": {"primary": "&H000000FF&", "outline": 5, "size": 70},
    "minimal": {"primary": "&H00FFFFFF&", "outline": 2, "size": 52},
    "classic": {"primary": "&H00FFFFFF&", "outline": 3, "size": 58},
}


class CreateJobRequest(BaseModel):
    source_url: str = Field(min_length=5, max_length=2048)
    captions_enabled: bool = True
    caption_style: str = "hormozi"
    clip_count: int = Field(default=3, ge=1, le=MAX_CLIPS)
    clip_duration: int = Field(default=30, ge=10, le=60)


class AdjustClipRequest(BaseModel):
    start_delta: float = Field(ge=-30, le=30)
    end_delta: float = Field(ge=-30, le=30)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_jobs() -> None:
    with jobs_lock:
        temp_file = JOBS_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_file.replace(JOBS_FILE)


def load_jobs() -> None:
    if not JOBS_FILE.exists():
        return
    try:
        jobs.update(json.loads(JOBS_FILE.read_text(encoding="utf-8")))
        for job in jobs.values():
            if job.get("status") in {"queued", "downloading", "analyzing", "transcribing", "finding_highlights", "generating_captions", "rendering", "finalizing"}:
                job.update(status="failed", current_stage="failed", current_step="Server restarted", message="The server restarted before this job completed.", error={"code": "SERVER_RESTARTED", "message": "Please submit the video again."}, updated_at=now())
        save_jobs()
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not load persisted jobs: {error}")


def update_job(job_id: str, *, status: str | None = None, stage: str | None = None, progress: float | None = None, message: str | None = None, **extra: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        if status is not None:
            job["status"] = status
        if stage is not None:
            job["current_stage"] = stage
            job["current_step"] = message or stage.replace("_", " ").title()
        if progress is not None:
            job["progress"] = max(0.0, min(100.0, round(progress, 1)))
        if message is not None:
            job["message"] = message
            job["current_step"] = message
        job.update(extra)
        job["updated_at"] = now()
        save_jobs()


def fail_job(job_id: str, code: str, message: str, details: str = "") -> None:
    print(f"[{job_id}] {code}: {details or message}")
    update_job(job_id, status="failed", stage="failed", message=message, error={"code": code, "message": message})


def require_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "The requested job does not exist."})
        return jobs[job_id]


def new_job(source_type: str, source: str, request: CreateJobRequest) -> dict[str, Any]:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = {"id": job_id, "job_id": job_id, "source_type": source_type, "source": source, "status": "queued", "progress": 0.0, "current_stage": "queued", "current_step": "Queued", "message": "Waiting for an available worker.", "created_at": now(), "updated_at": now(), "captions_enabled": request.captions_enabled, "caption_style": request.caption_style if request.caption_style in CAPTION_STYLES else "hormozi", "clip_count": request.clip_count, "clip_duration": request.clip_duration, "cancel_requested": False, "clips": [], "error": None}
    with jobs_lock:
        jobs[job_id] = job
        save_jobs()
    executor.submit(process_job, job_id)
    return job


@app.on_event("startup")
def startup() -> None:
    load_jobs()
    print(f"{APP_NAME} ready; worker concurrency={MAX_WORKERS}, output={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_NAME}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Backend is alive!", "service": APP_NAME}


@app.post("/api/jobs", status_code=202)
def create_job(request: CreateJobRequest) -> dict[str, Any]:
    if not request.source_url.startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail={"code": "INVALID_URL", "message": "Enter a valid HTTP(S) video URL."})
    return new_job("url", request.source_url, request)


@app.post("/api/jobs/upload", status_code=202)
async def upload_job(file: UploadFile = File(...), captions_enabled: bool = True, caption_style: str = "hormozi", clip_count: int = 3, clip_duration: int = 30) -> dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        raise HTTPException(status_code=400, detail={"code": "UNSUPPORTED_FORMAT", "message": "Upload MP4, MOV, MKV, WebM, or AVI video."})
    request = CreateJobRequest(captions_enabled=captions_enabled, caption_style=caption_style, clip_count=min(MAX_CLIPS, max(1, clip_count)), clip_duration=min(60, max(10, clip_duration)), source_url="upload")
    destination = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail={"code": "FILE_TOO_LARGE", "message": f"Video must be smaller than {MAX_UPLOAD_MB} MB."})
                output.write(chunk)
    finally:
        await file.close()
    return new_job("upload", str(destination), request)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return require_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    if job["status"] not in {"completed", "failed", "cancelled"}:
        update_job(job_id, status="cancelled", stage="cancelled", message="Cancelled by user.", cancel_requested=True)
    return require_job(job_id)


@app.get("/api/clips/{filename}")
def get_clip(filename: str):
    path = (OUTPUTS_DIR / Path(filename).name).resolve()
    if not path.is_file() or path.parent != OUTPUTS_DIR:
        raise HTTPException(status_code=404, detail={"code": "CLIP_NOT_FOUND", "message": "The generated clip is unavailable."})
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/thumbnails/{filename}")
def get_thumbnail(filename: str):
    path = (THUMBNAILS_DIR / Path(filename).name).resolve()
    if not path.is_file() or path.parent != THUMBNAILS_DIR:
        raise HTTPException(status_code=404, detail={"code": "THUMBNAIL_NOT_FOUND", "message": "The thumbnail is unavailable."})
    return FileResponse(path, media_type="image/jpeg")


def run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{label} failed: {error.stderr[-1000:]}") from error


def probe(path: Path) -> dict[str, Any]:
    result = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], "Video validation")
    data = json.loads(result.stdout)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("The source contains no video stream.")
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise RuntimeError("The source has an invalid duration.")
    return {"duration": duration, "width": int(video["width"]), "height": int(video["height"]), "has_audio": any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))}


def get_whisper() -> WhisperModel:
    global whisper_model
    if whisper_model is None:
        with whisper_lock:
            if whisper_model is None:
                print(f"Loading Faster-Whisper {WHISPER_MODEL} on {WHISPER_DEVICE}")
                whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return whisper_model


def detect_scene_changes(source: Path, threshold: float = 0.3) -> list[float]:
    """Detect visual scene-cut timestamps so clips can start on a clean cut
    instead of mid-shot. Downscales first to keep this fast."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(source), "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return []
    times = []
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            match = re.search(r"pts_time:(\d+\.?\d*)", line)
            if match:
                times.append(float(match.group(1)))
    return sorted(times)


def transcribe(source: Path, work_dir: Path) -> list[dict[str, Any]]:
    audio = work_dir / "audio.wav"
    run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(audio)], "Audio extraction")
    segments, _ = get_whisper().transcribe(str(audio), word_timestamps=True, beam_size=1, vad_filter=True, condition_on_previous_text=False)
    transcript = []
    for segment in segments:
        words = [{"start": word.start, "end": word.end, "word": word.word.strip()} for word in (segment.words or [])]
        transcript.append({"start": segment.start, "end": segment.end, "text": segment.text.strip(), "words": words})
    return transcript


def choose_highlights(transcript: list[dict[str, Any]], duration: float, count: int, clip_duration: int, scene_times: list[float] | None = None) -> list[dict[str, Any]]:
    """Use OpenAI when configured; otherwise use a reliable local fallback.
    When scene_times is given, clip start times snap to the nearest visual
    scene-cut (within 3s) so clips begin on a clean cut."""
    scene_times = scene_times or []

    def snap(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for clip in clips:
            length = clip["end"] - clip["start"]
            if scene_times:
                nearest = min(scene_times, key=lambda t: abs(t - clip["start"]))
                if abs(nearest - clip["start"]) <= 3:
                    clip["start"] = round(max(0.0, nearest), 2)
                    clip["end"] = round(min(duration, clip["start"] + length), 2)
        clips.sort(key=lambda clip: clip["start"])
        deduped: list[dict[str, Any]] = []
        for clip in clips:
            if not deduped or clip["start"] >= deduped[-1]["end"]:
                deduped.append(clip)
        return deduped

    if OPENAI_API_KEY and transcript:
        source_text = "\n".join(f"[{segment['start']:.1f}-{segment['end']:.1f}] {segment['text']}" for segment in transcript)
        if len(source_text) > 12000:
            source_text = source_text[:12000] + "\n...[transcript truncated]"
        prompt = f"""You are an expert short-video editor who finds the most viral, attention-grabbing moments in a video transcript.

Pick exactly {count} non-overlapping highlights. Each highlight must be between 10 and {clip_duration} seconds long. Prioritize moments with: strong emotion, a punchline or twist, a bold or surprising claim, a clear hook in the first 2 seconds, or a satisfying payoff.

Preserve the transcript's original spoken language exactly; never translate Hindi, Hinglish, Gujarati, or English.

Return JSON only in this exact shape:
{{"highlights":[{{"start":number,"end":number,"tag":string,"reason":string}}]}}

Transcript:
{source_text}"""
        payload = json.dumps({"model": OPENAI_MODEL, "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}).encode()
        request = Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, method="POST")
        for attempt in range(2):
            try:
                with urlopen(request, timeout=45) as response:
                    content = json.loads(json.loads(response.read())["choices"][0]["message"]["content"])
                candidates = []
                for item in content.get("highlights", []):
                    start = max(0.0, float(item["start"]))
                    end = min(duration, min(float(item["end"]), start + clip_duration))
                    if end - start >= 10:
                        candidates.append({"start": round(start, 2), "end": round(end, 2), "title": "", "tag": str(item.get("tag", "HIGHLIGHT"))[:32]})
                selected = snap(candidates)
                if selected:
                    return selected[:count]
                break
            except (URLError, TimeoutError, OSError) as error:
                print(f"OpenAI attempt {attempt + 1} failed: {error}")
                if attempt == 0:
                    time.sleep(2)
                    continue
            except (KeyError, TypeError, ValueError) as error:
                print(f"OpenAI returned an unusable response: {error}")
                break
    # Deterministic fallback preserves the spoken language and works if an AI provider is unavailable.
    if not transcript:
        windows = [{"start": round(index * max(1, duration - clip_duration) / max(1, count - 1), 1), "end": round(min(duration, index * max(1, duration - clip_duration) / max(1, count - 1) + clip_duration), 1), "title": f"Highlight {index + 1}", "tag": "HIGHLIGHT"} for index in range(count)]
        return snap(windows)[:count] or windows[:count]
    windows = []
    for index in range(count):
        target = duration * (index + .5) / count
        nearest = min(transcript, key=lambda segment: abs(segment["start"] - target))
        start = max(0.0, min(duration - clip_duration, nearest["start"] - 2))
        end = min(duration, start + clip_duration)
        title = re.sub(r"[\s,.!?;:]+$", "", nearest["text"])[:70] or f"Highlight {index + 1}"
        windows.append({"start": round(start, 2), "end": round(end, 2), "title": title, "tag": "HIGHLIGHT"})
    return snap(windows)[:count] or windows[:count]


def face_center(source: Path) -> float:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened() or face_cascade.empty():
        return .5
    frames, width, centers = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))), max(1, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))), []
    for position in (.15, .5, .85):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frames * position))
        ok, frame = capture.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, _, w, _ in face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5):
                centers.append((x + w / 2) / width)
    capture.release()
    return min(.85, max(.15, sum(centers) / len(centers))) if centers else .5


def ass_time(value: float) -> str:
    value = max(0, value)
    return f"{int(value // 3600)}:{int(value % 3600 // 60):02d}:{value % 60:05.2f}"


def write_ass(segments: list[dict[str, Any]], clip_start: float, path: Path, style_key: str) -> None:
    style = CAPTION_STYLES.get(style_key, CAPTION_STYLES["hormozi"])
    header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: {OUTPUT_WIDTH}\nPlayResY: {OUTPUT_HEIGHT}\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Default,Arial,{style['size']},{style['primary']},&H000000FF&,&H00000000&,&H66000000&,1,0,0,0,100,100,0,0,1,{style['outline']},0,2,50,50,260,1\nStyle: Emphasis,Arial,{style['size'] + 8},&H0000FFFF&,&H000000FF&,&H00000000&,&H66000000&,1,0,0,0,100,100,0,0,1,{style['outline'] + 1},0,2,50,50,260,1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    lines = [header]
    for segment in segments:
        words = segment.get("words") or [{"start": segment["start"], "end": segment["end"], "word": segment["text"]}]
        for word in words:
            text = word["word"].strip().upper().replace("{", "\\{").replace("}", "\\}")
            if text:
                start, end = word["start"] - clip_start, word["end"] - clip_start
                if end > 0:
                    style_name = "Emphasis" if len(text) >= 6 or "!" in text else "Default"
                    lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(max(start + 0.1, end))},{style_name},,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")


def render_clip(source: Path, metadata: dict[str, Any], clip: dict[str, Any], transcript: list[dict[str, Any]], job: dict[str, Any], index: int, work_dir: Path) -> tuple[Path, Path]:
    output, thumbnail, ass = OUTPUTS_DIR / f"{job['id']}_{index}.mp4", THUMBNAILS_DIR / f"{job['id']}_{index}.jpg", work_dir / f"{index}.ass"
    selected = [segment for segment in transcript if segment["end"] > clip["start"] and segment["start"] < clip["end"]]
    if metadata["height"] >= metadata["width"]:
        filters = [f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih):black"]
    else:
        filters = [f"crop=ih*9/16:ih:'max(0,min(iw-ih*9/16,iw*{job['face_center']:.4f}-ih*9/32))':0,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos"]
    if job["captions_enabled"] and selected:
        write_ass(selected, clip["start"], ass, job["caption_style"])
        filters.append("subtitles='" + str(ass.resolve()).replace("\\", "/").replace(":", "\\:") + "'")
    command = ["ffmpeg", "-y", "-ss", str(clip["start"]), "-i", str(source), "-t", str(clip["end"] - clip["start"]), "-vf", ",".join(filters), "-c:v", VIDEO_ENCODER]
    if VIDEO_ENCODER == "libx264":
        command += ["-preset", VIDEO_PRESET, "-crf", os.getenv("VIDEO_CRF", "20")]
    command += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
    run(command, "Clip rendering")
    run(["ffmpeg", "-y", "-ss", "0.1", "-i", str(output), "-frames:v", "1", "-q:v", "2", str(thumbnail)], "Thumbnail generation")
    validate_output(output)
    return output, thumbnail


def validate_output(path: Path) -> None:
    info = probe(path)
    if path.stat().st_size < 1024 or info["duration"] <= .5 or not info["has_audio"]:
        raise RuntimeError("Rendered output validation failed.")


def process_job(job_id: str) -> None:
    started, work_dir, source = time.monotonic(), TEMP_DIR / job_id, None
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        job = require_job(job_id)
        if job["cancel_requested"]:
            return
        if job["source_type"] == "url":
            update_job(job_id, status="downloading", stage="downloading", progress=5, message="Downloading source video.")
            source = SOURCES_DIR / f"{job_id}.mp4"
            try:
                with yt_dlp.YoutubeDL({"outtmpl": str(source), "format": "bv*[height<=1080]+ba/b[height<=1080]/b", "merge_output_format": "mp4", "retries": 3, "fragment_retries": 3, "socket_timeout": 30, "quiet": True, "noplaylist": True}) as downloader:
                    downloader.download([job["source"]])
            except Exception as error:
                fail_job(job_id, "VIDEO_DOWNLOAD_FAILED", "Unable to download the YouTube video. It may be private, restricted, or temporarily blocked by YouTube.", str(error))
                return
        else:
            source = Path(job["source"])
        if require_job(job_id)["cancel_requested"]:
            return
        update_job(job_id, status="analyzing", stage="analyzing", progress=15, message="Validating video streams and framing.")
        metadata = probe(source)
        update_job(job_id, status="transcribing", stage="transcribing", progress=25, message="Transcribing speech and lyrics.")
        transcription_started = time.monotonic()
        transcript = transcribe(source, work_dir) if metadata["has_audio"] else []
        update_job(job_id, progress=40, transcript=transcript, metrics={"transcription_seconds": round(time.monotonic() - transcription_started, 2)})
        if require_job(job_id)["cancel_requested"]:
            return
        update_job(job_id, progress=45, message="Detecting scene changes.")
        scene_times = detect_scene_changes(source)
        update_job(job_id, status="finding_highlights", stage="finding_highlights", progress=50, message="Selecting the strongest moments.")
        highlights = choose_highlights(transcript, metadata["duration"], job["clip_count"], job["clip_duration"], scene_times)
        update_job(job_id, status="generating_captions", stage="generating_captions", progress=55, message="Preparing synchronized captions.", highlights=highlights)
        job = require_job(job_id)
        job["face_center"] = face_center(source) if metadata["width"] > metadata["height"] else .5
        update_job(job_id, status="rendering", stage="rendering", progress=60, message=f"Rendering 0 of {len(highlights)} clips.")
        results, render_started = [], time.monotonic()
        for index, clip in enumerate(highlights):
            if require_job(job_id)["cancel_requested"]:
                return
            output, thumbnail = render_clip(source, metadata, clip, transcript, require_job(job_id), index, work_dir)
            selected = [segment for segment in transcript if segment["end"] > clip["start"] and segment["start"] < clip["end"]]
            results.append({"id": f"clip_{index}", "title": f"Highlight {index + 1}", "moment_tag": clip.get("tag", "HIGHLIGHT"), "video_url": f"/api/clips/{output.name}", "thumbnail_url": f"/api/thumbnails/{thumbnail.name}", "duration_ms": int((clip["end"] - clip["start"]) * 1000), "fluff_cut_percent": 0, "captions": [{"text": s["text"], "start_ms": max(0, int((s["start"] - clip["start"]) * 1000)), "end_ms": max(0, int((s["end"] - clip["start"]) * 1000))} for s in selected]})
            update_job(job_id, progress=60 + 35 * (index + 1) / len(highlights), message=f"Rendering {index + 1} of {len(highlights)} clips.")
        total = round(time.monotonic() - started, 2)
        update_job(job_id, status="finalizing", stage="finalizing", progress=98, message="Validating final outputs.")
        update_job(job_id, status="completed", stage="completed", progress=100, message="Your Shorts are ready.", clips=results, result=results, metrics={**require_job(job_id).get("metrics", {}), "rendering_seconds": round(time.monotonic() - render_started, 2), "total_seconds": total})
        print(f"[{job_id}] completed in {total}s")
    except Exception as error:
        fail_job(job_id, "PROCESSING_FAILED", "Video processing failed. Please try another video.", str(error))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/clips/{job_id}/{clip_index}/adjust")
def adjust_clip(job_id: str, clip_index: int, request: AdjustClipRequest) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail={"code": "EDITOR_NOT_AVAILABLE", "message": "Clip timing edits are not enabled in this version."})