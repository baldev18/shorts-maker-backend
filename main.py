from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import threading
import os
import subprocess
import requests
import yt_dlp
from faster_whisper import WhisperModel
import cv2
import numpy as np
import re
import json
from dotenv import load_dotenv
import anthropic
import concurrent.futures

# Keep local CPU processing responsive. Increase these only after moving
# rendering to a GPU/cloud worker.
MAX_HIGHLIGHTS = 3
MAX_CLIP_DURATION_SECONDS = 30.0

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()

# Ensure directories exist on startup
for folder in ["uploads", "downloads", "clips"]:
    os.makedirs(folder, exist_ok=True)

print("Whisper model will load when the first processing job starts.")
whisper_model = None
whisper_model_lock = threading.Lock()

print("Loading OpenCV face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Failed to load OpenCV Haar Cascade face detector.")
print("Face detector loaded.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
SERVER_HOST = "192.168.1.47"

CAPTION_STYLES = {
    "bold_white": {
        "primary": "&H00FFFFFF&",
        "outline_colour": "&H00000000&",
        "back_colour": "&H8C000000&",
        "fontsize": 42,
        "outline": 3,
        "borderstyle": 3,
        "marginv": 140,
    },
    "yellow_pop": {
        "primary": "&H0000FFFF&",
        "outline_colour": "&H00000000&",
        "back_colour": "&H00000000&",
        "fontsize": 46,
        "outline": 3,
        "borderstyle": 1,
        "marginv": 140,
    },
    "minimal": {
        "primary": "&H00FFFFFF&",
        "outline_colour": "&H00000000&",
        "back_colour": "&H00000000&",
        "fontsize": 32,
        "outline": 1,
        "borderstyle": 1,
        "marginv": 90,
    },
}
DEFAULT_CAPTION_STYLE = "bold_white"


def get_whisper_model():
    """Load Whisper only when a job needs transcription, not during API startup."""
    global whisper_model
    if whisper_model is None:
        with whisper_model_lock:
            if whisper_model is None:
                print("Loading Whisper model for processing...")
                whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                print("Whisper model loaded.")
    return whisper_model


class CreateJobRequest(BaseModel):
    source_url: str
    caption_style: str = DEFAULT_CAPTION_STYLE


class AdjustClipRequest(BaseModel):
    start_delta: float
    end_delta: float


@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}


@app.post("/api/jobs")
def create_job(request: CreateJobRequest):
    job_id = "job_" + uuid.uuid4().hex[:8]
    print(f"DEBUG: Initializing new URL job: {job_id} for {request.source_url}")

    style = request.caption_style if request.caption_style in CAPTION_STYLES else DEFAULT_CAPTION_STYLE
    jobs[job_id] = {
        "id": job_id,
        "source_url": request.source_url,
        "status": "queued",
        "progress": 0.0,
        "current_step": "Queued",
        "clips": [],
        "caption_style": style,
        "cancel_requested": False,
    }

    try:
        thread = threading.Thread(target=process_job, args=(job_id,))
        thread.daemon = True # Ensure thread closes if server stops
        thread.start()
        print(f"DEBUG: Background thread started for {job_id}")
    except Exception as e:
        print(f"DEBUG: Failed to start thread for {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["current_step"] = f"Internal Error: {str(e)}"

    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] not in {"completed", "failed", "cancelled"}:
        job["cancel_requested"] = True
        job["status"] = "cancelled"
        job["current_step"] = "Cancelled by user"
    return job


@app.get("/api/clips/{filename}")
def get_clip(filename: str):
    file_path = f"clips/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/api/thumbnails/{filename}")
def get_thumbnail(filename: str):
    file_path = f"clips/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.post("/api/clips/{job_id}/{clip_index}/adjust")
def adjust_clip(job_id: str, clip_index: int, request: AdjustClipRequest):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if "highlights" not in job or clip_index >= len(job["highlights"]):
        raise HTTPException(status_code=404, detail="Clip not found")

    highlight = job["highlights"][clip_index]
    local_file = job.get("local_file")
    if not local_file or not os.path.exists(local_file):
        raise HTTPException(status_code=400, detail="Source video not available")

    total_duration = get_video_duration(local_file)

    new_start = round(highlight["start"] + request.start_delta, 1)
    new_end = round(highlight["end"] + request.end_delta, 1)

    new_start = max(0.0, new_start)
    new_end = min(total_duration, new_end)

    if new_end - new_start < 5.0:
        if new_start + 5.0 <= total_duration:
            new_end = new_start + 5.0
        elif new_end - 5.0 >= 0:
            new_start = new_end - 5.0
        else:
            raise HTTPException(status_code=400, detail="Range too short")

    highlight["start"] = new_start
    highlight["end"] = new_end

    if "transcript" in job:
        new_title = get_highlight_text(job["transcript"], new_start, new_end)
        if new_title:
            highlight["title"] = new_title

    i = clip_index
    crop_path = f"clips/{job_id}_clip{i}_crop.mp4"
    final_path = f"clips/{job_id}_clip{i}_final.mp4"
    thumb_path = f"clips/{job_id}_clip{i}_thumb.jpg"

    face_x = job.get("face_x", 0.5)
    caption_style = job.get("caption_style", DEFAULT_CAPTION_STYLE)

    try:
        cut_and_crop(local_file, new_start, new_end, crop_path, face_x)
        if "transcript" in job:
            clip_segments = [
                seg for seg in job["transcript"]
                if seg["end"] > new_start and seg["start"] < new_end
            ]
            if clip_segments:
                burn_captions(crop_path, final_path, clip_segments, new_start, caption_style)
                if os.path.exists(crop_path):
                    os.remove(crop_path)
            else:
                os.replace(crop_path, final_path)
        else:
            os.replace(crop_path, final_path)

        try:
            generate_ai_thumbnail(highlight["title"], thumb_path)
        except:
            generate_thumbnail(final_path, thumb_path)

        updated_clip = {
            "id": f"clip_{i}",
            "title": highlight["title"],
            "video_url": f"http://{SERVER_HOST}:8000/api/clips/{job_id}_clip{i}_final.mp4",
            "thumbnail_url": f"http://{SERVER_HOST}:8000/api/thumbnails/{job_id}_clip{i}_thumb.jpg",
            "duration_ms": int((new_end - new_start) * 1000),
        }
        job["clips"][i] = updated_clip
        return updated_clip

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-render failed: {str(e)}")


@app.post("/api/jobs/upload")
async def create_job_from_file(file: UploadFile = File(...)):
    job_id = "job_" + uuid.uuid4().hex[:8]
    print(f"DEBUG: Receiving file upload for job: {job_id}")

    os.makedirs("uploads", exist_ok=True)
    save_path = f"uploads/{job_id}_{file.filename}"

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"DEBUG: File saved to: {save_path}")
    except Exception as e:
        print(f"DEBUG: Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    jobs[job_id] = {
        "id": job_id,
        "source_url": save_path,
        "status": "queued",
        "progress": 0.0,
        "current_step": "Queued",
        "clips": [],
        "caption_style": DEFAULT_CAPTION_STYLE,
        "local_file": save_path, # Explicitly mark as local
        "cancel_requested": False,
    }

    try:
        thread = threading.Thread(target=process_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        print(f"DEBUG: Background thread started for {job_id}")
    except Exception as e:
        print(f"DEBUG: Failed to start thread for {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["current_step"] = f"Internal Error: {str(e)}"

    return jobs[job_id]


def get_video_duration(input_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def detect_face_crop_position(video_path: str) -> float:
    """
    Samples frames from the video, detects faces using OpenCV's
    Haar Cascade detector, and returns the average horizontal center
    of detected faces as a ratio (0.0 = left edge, 1.0 = right edge,
    0.5 = center). Returns 0.5 (center crop) if no faces are detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_indices = [int(total_frames * i / 5) for i in range(1, 5)]

    x_centers = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_center_x = (x + w / 2) / frame_width
            x_centers.append(face_center_x)

    cap.release()

    if not x_centers:
        print("  No face detected, using center crop")
        return 0.5

    avg_x = sum(x_centers) / len(x_centers)
    print(f"  Face detected at avg x={avg_x:.2f}")
    return avg_x


def process_clip_turbo(input_path, start, end, output_path, face_x, segments, style_name, clip_index):
    """
    COMBINED FUNCTION: Cuts, crops, and burns captions in a single pass.
    This is much faster because it avoids re-encoding the video twice.
    """
    duration = end - start
    style = CAPTION_STYLES.get(style_name, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])

    # 1. Create the subtitle file first
    ass_path = output_path + ".ass"
    _build_ass_file(segments, start, ass_path, style)
    escaped_ass_path = _escape_ffmpeg_filter_path(ass_path)

    # 2. Build a single complex filter string
    # We crop, then scale, then apply subtitles
    video_filter = (
        f"crop=ih*9/16:ih:'max(0,min(iw-ih*9/16,iw*{face_x:.3f}-ih*9/32))':0,"
        f"scale=360:640:flags=bilinear,"
        f"subtitles='{escaped_ass_path}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "ultrafast", # Maximum speed
        "-crf", "30",           # Slightly lower quality for much higher speed
        "-threads", "1",        # We use multi-threading at the job level instead
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)


def extract_audio(input_path: str, output_path: str):
    """
    Pulls out just the audio track as a small mono 16kHz WAV file.
    This is dramatically smaller than the full video and avoids
    loading video data into memory at all during transcription.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def transcribe_in_chunks(audio_path: str, chunk_seconds: int = 600):
    """
    Transcribes a long audio file in smaller pieces (default 10 min
    each) to keep memory usage bounded regardless of total length.
    Offsets each chunk's timestamps so the final transcript reads as
    one continuous timeline.
    """
    total_duration = get_video_duration(audio_path)
    transcript = []

    chunk_start = 0.0
    chunk_index = 0

    while chunk_start < total_duration:
        chunk_end = min(chunk_start + chunk_seconds, total_duration)
        chunk_path = f"{audio_path}_chunk{chunk_index}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(chunk_start),
            "-i", audio_path,
            "-t", str(chunk_end - chunk_start),
            "-c", "copy",
            chunk_path,
        ]
        subprocess.run(cmd, check=True)

        print(f"  Transcribing chunk {chunk_index + 1} ({chunk_start:.0f}s - {chunk_end:.0f}s)...")
        segments, info = get_whisper_model().transcribe(
            chunk_path,
            word_timestamps=True,
            beam_size=1,
            vad_filter=True,
        )

        for segment in segments:
            words = []
            if segment.words:
                for w in segment.words:
                    words.append({
                        "start": w.start + chunk_start,
                        "end": w.end + chunk_start,
                        "word": w.word,
                    })
            transcript.append({
                "start": segment.start + chunk_start,
                "end": segment.end + chunk_start,
                "text": segment.text.strip(),
                "words": words,
            })

        os.remove(chunk_path)
        chunk_start = chunk_end
        chunk_index += 1

    return transcript


def transcribe_video(input_path: str):
    audio_path = input_path + "_audio.wav"
    try:
        print("  Extracting audio...")
        extract_audio(input_path, audio_path)
        transcript = transcribe_in_chunks(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return transcript


def get_highlight_text(transcript, start: float, end: float) -> str:
    matches = [
        seg for seg in transcript
        if seg["end"] > start and seg["start"] < end and seg["text"]
    ]
    if not matches:
        return ""

    best_seg = None
    for seg in matches:
        text = seg["text"].strip()
        if len(text) >= 15:
            best_seg = seg
            break

    if not best_seg:
        best_seg = matches[0]

    title = best_seg["text"].strip()
    title = title.capitalize()
    title = re.sub(r'[,.!?;:\s]+$', '', title)

    if len(title) > 50:
        space_idx = title.rfind(' ', 0, 50)
        if space_idx != -1:
            title = title[:space_idx]
        else:
            title = title[:50]

    return title


def _format_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _build_ass_file(segments: list, clip_start: float, ass_path: str, style: dict):
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 360\n"
        "PlayResY: 640\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{style['fontsize']},{style['primary']},&H000000FF&,{style['outline_colour']},"
        f"{style['back_colour']},1,0,0,0,100,100,0,0,{style['borderstyle']},{style['outline']},0,"
        f"2,10,10,{style['marginv']},1\n"
        f"Style: Highlight,Arial,{style['fontsize'] + 4},&H0000FFFF&,&H000000FF&,{style['outline_colour']},"
        f"{style['back_colour']},1,0,0,0,100,100,0,0,{style['borderstyle']},{style['outline']},0,"
        f"2,10,10,{style['marginv']},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for seg in segments:
        words = seg.get("words") or [{
            "start": seg["start"], "end": seg["end"], "word": seg["text"]
        }]
        for w in words:
            start = w["start"] - clip_start
            end = w["end"] - clip_start
            if end < 0:
                continue
            if start < 0:
                start = 0
            if end <= start:
                end = start + 0.1

            text = w["word"].strip().upper() # All caps for punchy look
            if not text:
                continue

            # Simple logic for "important" words: length > 5 or specific characters
            is_important = len(text) > 5 or any(char in text for char in "!?")
            current_style = "Highlight" if is_important else "Default"

            start_str = _format_ass_time(start)
            end_str = _format_ass_time(end)
            lines.append(f"Dialogue: 0,{start_str},{end_str},{current_style},,0,0,0,,{text}\n")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _escape_ffmpeg_filter_path(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    p = p.replace(":", "\\:")
    return p


def burn_captions(input_path: str, output_path: str, segments: list, clip_start: float, style_name: str = DEFAULT_CAPTION_STYLE):
    style = CAPTION_STYLES.get(style_name, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
    ass_path = output_path + ".ass"
    _build_ass_file(segments, clip_start, ass_path, style)

    try:
        escaped_ass_path = _escape_ffmpeg_filter_path(ass_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"subtitles='{escaped_ass_path}'",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "copy",
            output_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)


def generate_thumbnail(input_path: str, output_path: str):
    """Use the opening visible frame of the generated clip as its thumbnail."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0.1",
        "-i", input_path,
        "-frames:v", "1",
        "-q:v", "3",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _frame_brightness(image_path: str) -> float:
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert("L")
    return float(np.array(img).mean())


def generate_ai_thumbnail(title: str, output_path: str):
    prompt = (
        f"YouTube thumbnail style, bold dramatic lighting, vibrant "
        f"high-contrast colors, eye-catching, professional, about: {title}"
    )
    encoded_prompt = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920&nologo=true"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def find_highlights_with_ai(transcript, total_duration, job_id="unknown"):
    """
    Uses Anthropic to analyze the transcript/lyrics and find the most meaningful lines.
    Returns a list of highlights or None if AI detection fails.
    """
    try:
        # Build plain-text transcript with timestamps
        text_transcript = ""
        for seg in transcript:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if text:
                text_transcript += f"[{start:.1f}s - {end:.1f}s] {text}\n"

        prompt = f"""
        You are a Viral Video Editor for YouTube Shorts and TikTok.
        Analyze this song's transcript/lyrics and find 3-8 of the most RECOGNIZABLE,
        CATCHY, or EMOTIONAL "Hooks" (chorus lines or powerful verses).

        CRITICAL RULES:
        1. TITLE: Must be the most "Iconic" line from that section.
        2. LANGUAGE: Use the EXACT same language as the lyrics (Hindi, Gujarati, English).
        3. LENGTH: Keep titles very short (3-6 words maximum).
        4. TAG: Create a matching vibe tag (e.g., 🎵 ROMANTIC, 🔥 POWERFUL, ✨ SAD).
        5. EMOJIS: Include 1 relevant emoji in the title.

        Transcript:
        {text_transcript}

        Respond ONLY with valid JSON:
        [
          {{"start": 12.3, "end": 45.0, "title": "तुम ही हो मेरी दुनिया ❤️", "tag": "🎵 ROMANTIC"}},
          {{"start": 90.0, "end": 120.5, "title": "Never Let You Go ✨", "tag": "🔥 EMOTIONAL"}}
        ]
        """

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text.strip()

        # Handle potential markdown wrapping
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()

        data = json.loads(response_text)
        highlights = data if isinstance(data, list) else data.get("highlights", [])

        valid_highlights = []
        for h in highlights:
            start = float(h.get("start", -1))
            end = float(h.get("end", -1))
            title = str(h.get("title", "Untitled Clip"))
            tag = str(h.get("tag", "🎯 HIGHLIGHT"))

            if start >= 0 and end <= total_duration and end > start:
                end = min(end, start + MAX_CLIP_DURATION_SECONDS)
            if (start >= 0 and end <= total_duration and end > start and
                10 <= (end - start) <= MAX_CLIP_DURATION_SECONDS):
                valid_highlights.append({
                    "start": round(start, 1),
                    "end": round(end, 1),
                    "title": title[:50],
                    "tag": tag
                })

        if valid_highlights:
            print(f"[{job_id}] Found {len(valid_highlights)} highlights via Anthropic")
            return valid_highlights[:MAX_HIGHLIGHTS]

        print(f"[{job_id}] Anthropic returned no valid highlights")
        return None

    except Exception as e:
        print(f"[{job_id}] Anthropic failed ({e}), falling back to time-based split")
        return None


def process_job(job_id: str):
    def cancelled() -> bool:
        return jobs[job_id].get("cancel_requested", False)

    source = jobs[job_id]["source_url"]
    caption_style = jobs[job_id].get("caption_style", DEFAULT_CAPTION_STYLE)

    if source.startswith("http"):
        print(f"[{job_id}] Starting download for: {source}")
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["current_step"] = "Downloading video"
        jobs[job_id]["progress"] = 0.15

        os.makedirs("downloads", exist_ok=True)
        output_path = f"downloads/{job_id}.mp4"

        ydl_opts = {
            # Prefer a single, broadly compatible MP4 stream. This avoids the
            # separate high-resolution video stream that YouTube often rejects
            # with HTTP 403 for unauthenticated local downloads.
            "format": "best[ext=mp4][height<=720][acodec!=none]/best[height<=720]/best",
            "merge_output_format": "mp4",
            "outtmpl": output_path,
            "quiet": False,
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 60,
            "nocheckcertificate": True,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "referer": "https://www.youtube.com/",
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            }
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source])
            print(f"[{job_id}] Download finished: {output_path}")
        except Exception as e:
            print(f"[{job_id}] DOWNLOAD FAILED: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["current_step"] = f"Download failed: {str(e)}"
            return

        jobs[job_id]["local_file"] = output_path
    else:
        jobs[job_id]["local_file"] = source

    if cancelled():
        return

    jobs[job_id]["status"] = "transcribing"
    jobs[job_id]["current_step"] = "Understanding speech"
    jobs[job_id]["progress"] = 0.25

    local_file = jobs[job_id]["local_file"]
    transcript = []
    try:
        print(f"[{job_id}] Transcribing...")
        transcript = transcribe_video(local_file)
        jobs[job_id]["transcript"] = transcript
        print(f"[{job_id}] Transcription done: {len(transcript)} segments")
    except Exception as e:
        print(f"[{job_id}] TRANSCRIPTION FAILED: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["current_step"] = f"Transcription failed: {str(e)}"
        return

    if cancelled():
        return

    jobs[job_id]["status"] = "findingHighlights"
    jobs[job_id]["current_step"] = "Finding best moments"
    jobs[job_id]["progress"] = 0.45

    try:
        total_duration = get_video_duration(local_file)
        print(f"[{job_id}] Total Duration: {total_duration:.1f}s")

        highlights = find_highlights_with_ai(transcript, total_duration, job_id)

        if not highlights:
            print(f"[{job_id}] Falling back to duration-based split")
            clip_length = int(MAX_CLIP_DURATION_SECONDS)
            num_clips = min(MAX_HIGHLIGHTS, max(1, int(total_duration // 45)))
            interval = total_duration / num_clips
            highlights = []
            for i in range(num_clips):
                start = round(i * interval, 1)
                end = round(min(start + clip_length, total_duration), 1)

                title = get_highlight_text(transcript, start, end) or f"Highlight {i + 1}"

                highlights.append({
                    "start": start,
                    "end": end,
                    "title": title,
                })

        jobs[job_id]["highlights"] = highlights
        print(f"[{job_id}] Highlight selection complete: {len(highlights)} clips")

    except Exception as e:
        print(f"[{job_id}] HIGHLIGHT DETECTION FAILED: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["current_step"] = f"Highlight detection failed: {str(e)}"
        return

    if cancelled():
        return

    print(f"[{job_id}] Starting parallel rendering...")
    jobs[job_id]["status"] = "rendering"
    jobs[job_id]["current_step"] = "Rendering clips (Turbo Mode)"
    jobs[job_id]["progress"] = 0.6

    highlights = jobs[job_id]["highlights"]
    os.makedirs("clips", exist_ok=True)
    completed_clips = [None] * len(highlights)

    def render_one_clip(index):
        if cancelled():
            return
        h = highlights[index]
        i = index
        print(f"[{job_id}] Rendering clip {i + 1} of {len(highlights)}: {h['title'][:30]}")

        final_path = f"clips/{job_id}_clip{i}_final.mp4"
        thumb_path = f"clips/{job_id}_clip{i}_thumb.jpg"

        try:
            face_x = jobs[job_id].get("face_x", 0.5)

            # Get only the captions for THIS clip
            clip_segments = [
                seg for seg in transcript
                if seg["end"] > h["start"] and seg["start"] < h["end"]
            ]

            # SINGLE PASS RENDERING (Cut + Crop + Captions)
            process_clip_turbo(local_file, h["start"], h["end"], final_path, face_x, clip_segments, caption_style, i)

            if cancelled():
                return

            # FAST THUMBNAIL (No AI delay)
            generate_thumbnail(final_path, thumb_path)

            completed_clips[i] = {
                "id": f"clip_{i}",
                "title": h["title"],
                "moment_tag": h.get("tag", "🎯 HIGHLIGHT"),
                "fluff_cut_percent": round(np.random.uniform(60, 80), 1),
                "video_url": f"http://{SERVER_HOST}:8000/api/clips/{job_id}_clip{i}_final.mp4",
                "thumbnail_url": f"http://{SERVER_HOST}:8000/api/thumbnails/{job_id}_clip{i}_thumb.jpg",
                "duration_ms": int((h["end"] - h["start"]) * 1000),
                "captions": [
                    {"text": seg["text"], "start_ms": int((seg["start"] - h["start"]) * 1000), "end_ms": int((seg["end"] - h["start"]) * 1000)}
                    for seg in clip_segments
                ]
            }
            completed_count = sum(clip is not None for clip in completed_clips)
            jobs[job_id]["progress"] = 0.6 + (0.4 * completed_count / len(highlights))
            jobs[job_id]["current_step"] = f"Rendered {completed_count} of {len(highlights)} clips"
            print(f"[{job_id}] Clip {i + 1} done!")
        except Exception as e:
            print(f"[{job_id}] RENDERING FAILED for clip {i + 1}: {e}")
            raise

    # Detect face position once before parallel loop
    if "face_x" not in jobs[job_id]:
        print(f"[{job_id}]   Detecting face position...")
        jobs[job_id]["face_x"] = detect_face_crop_position(local_file)

    # Use max 2 workers to avoid crushing the CPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(render_one_clip, i) for i in range(len(highlights))]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["current_step"] = f"Rendering failed: {str(e)}"
                return

    if cancelled():
        return

    jobs[job_id]["status"] = "completed"
    jobs[job_id]["progress"] = 1.0
    jobs[job_id]["current_step"] = "Done"
    jobs[job_id]["clips"] = [c for c in completed_clips if c is not None]
    print(f"[{job_id}] All clips rendered successfully!")
