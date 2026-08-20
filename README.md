# Shorts Generator Backend

FastAPI API plus a bounded background worker for FFmpeg/Faster-Whisper video processing.

## Development

Install FFmpeg and FFprobe on your PATH, then:

```powershell
Copy-Item .env.example .env
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check: `http://127.0.0.1:8000/health`.

## Production

Set a real HTTPS domain in `CORS_ORIGINS`, copy `.env.example` to `.env`, then:

```bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/health
```

The Docker service restarts automatically after failures or host reboot. Put it behind an HTTPS reverse proxy such as Caddy, Nginx, Cloud Run, Render, Fly.io, or a managed container platform. Do not expose a home-PC IP to Play Store users.

## Operational notes

- Jobs are immediately accepted (`202`) and run in a single bounded worker by default.
- Job status is persisted in `data/jobs.json`; jobs interrupted by a restart become `SERVER_RESTARTED` failures rather than hanging forever.
- Final outputs and thumbnails are persisted under `data/`; transient audio/subtitle files are removed.
- The API returns relative output URLs. Flutter should resolve them against its configured API base URL.
- For GPU deployments set `WHISPER_DEVICE=cuda`, suitable `WHISPER_COMPUTE_TYPE`, and optionally `VIDEO_ENCODER=h264_nvenc` after installing CUDA/NVIDIA support.
