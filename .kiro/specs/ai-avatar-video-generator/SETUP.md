# AI Avatar Video Generator — Local Setup Guide

This guide covers everything needed to run the system locally. It is split into
what runs **without a GPU** (the full application skeleton) and what additionally
requires an **NVIDIA GPU + AI models** (actual video generation).

---

## 0. Prerequisites summary

| Component | Purpose | GPU needed? | Status on this machine |
|-----------|---------|-------------|------------------------|
| Python 3.12 | Backend runtime | No | ✅ installed |
| FFmpeg | Video/audio muxing | No | ✅ installed (winget, v9.0.1) |
| Redis | Task state + Celery broker | No | ✅ running (Docker `avatar-redis`) |
| celery, redis-py | Task queue | No | ✅ installed |
| FastAPI, uvicorn | REST + WebSocket API | No | ✅ installed |
| React + Vite | Frontend | No | ✅ installed (`avatar-frontend/`) |
| NVIDIA GPU (8GB+ VRAM) | AI inference | **Yes** | ❌ not present (`nvidia-smi` missing) |
| torch (CUDA build) | Model runtime | **Yes** | ⚠️ only CPU build installed |
| InsightFace | Face detection | Yes (practically) | ❌ not installed |
| GPT-SoVITS | Voice cloning | Yes | ❌ not installed |
| SadTalker | Lip sync | Yes | ❌ not installed |

---

## 1. Run the system WITHOUT a GPU (application skeleton)

This lets you start the API, worker, and frontend, upload materials, pass
validation, create tasks, and watch progress. Generation will stop at the voice
synthesis stage with a clear "model not installed" error — everything up to that
point is fully functional.

### 1.1 Environment config

Avatar settings are read from `.env.avatar` (kept separate from the workspace's
shared `.env`, which belongs to another project and uses a Docker-internal
`REDIS_URL`). The file is already created:

```
AVATAR_OUTPUT_DIR=C:\Users\ariel\OneDrive\Desktop\ai-avatar-output
REDIS_URL=redis://localhost:6379/0
MODEL_DIR=C:\Users\ariel\OneDrive\Desktop\ai-avatar-output\models
```

### 1.2 Install non-GPU Python deps

```powershell
python -m pip install -r requirements-avatar.txt
```

### 1.3 Start Redis (Docker)

```powershell
docker run -d --name avatar-redis -p 6379:6379 redis:7-alpine
# already running on this machine; re-use with: docker start avatar-redis
```

### 1.4 Install FFmpeg (already done here)

```powershell
winget install --id Gyan.FFmpeg -e
# open a new shell afterwards so the PATH update takes effect
```

### 1.5 Start the backend API

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/health  ->  {"status":"ok"}
```

### 1.6 Start the Celery worker

```powershell
python -m celery -A core.celery_app:celery_app worker --loglevel=info --pool=solo
# --pool=solo is recommended on Windows
# [tasks] should list: avatar.generate_video
```

### 1.7 Start the frontend

```powershell
cd avatar-frontend
npm install
npm run dev
# http://localhost:5173  (proxies /api and /api/v1/ws to :8000)
```

At this point the whole flow works except AI inference.

---

## 2. Enable ACTUAL video generation (requires NVIDIA GPU)

### 2.1 Hardware

An NVIDIA GPU with 8GB+ VRAM and current drivers. Verify with:

```powershell
nvidia-smi
```

### 2.2 Install the CUDA build of PyTorch

The currently installed torch is the CPU build. Replace it (pick the CUDA
version matching your driver, e.g. cu121):

```powershell
python -m pip uninstall -y torch torchaudio
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

### 2.3 Install the AI model packages

```powershell
python -m pip install insightface onnxruntime-gpu
```

InsightFace's `buffalo_l` pack downloads automatically on first use.

### 2.4 Install GPT-SoVITS (voice cloning)

Clone and install per the upstream instructions, then place weights under
`MODEL_DIR\gpt_sovits\`:

- Repo: https://github.com/RVC-Boss/GPT-SoVITS
- The `VoiceCloner` in `modules/voice_cloner.py` imports
  `GPT_SoVITS.inference.GPTSoVITSInference`, so that package must be importable.

### 2.5 Install SadTalker (lip sync)

Clone and install per the upstream instructions, then place checkpoints under
`MODEL_DIR\sadtalker\`:

- Repo: https://github.com/OpenTalker/SadTalker
- `modules/lip_sync_engine.py` loads the SadTalker checkpoints from that dir.

### 2.6 Expected model directory layout

```
<MODEL_DIR>\
  gpt_sovits\      # GPT-SoVITS weights
  sadtalker\       # SadTalker checkpoints
  models\          # InsightFace auto-download cache (buffalo_l)
```

### 2.7 Run the GPU end-to-end test

```powershell
$env:AVATAR_GPU_E2E = "1"
python -m pytest tests/avatar/integration/test_end_to_end.py -k FullPipeline -v
```

---

## 3. What was verified on this machine (no GPU)

- FFmpeg installed and on PATH (v9.0.1)
- Redis running and reachable (`PING True`)
- FastAPI backend live; `/health` and 404 handling confirmed
- Celery worker registered `avatar.generate_video`
- A real multipart submission returned `202` with a task id + summary
- The worker executed stage 1 (script processing, 10%) and then failed at stage 2
  (voice synthesis) with "GPT-SoVITS is not installed" — the exact, expected
  boundary without the AI models
- Frontend dev server live on :5173; Vite proxy routes to the backend

## 4. Known limitation

Without an NVIDIA GPU and the three AI model packages, generation cannot proceed
past voice synthesis. The application, API, task queue, progress streaming, and
frontend are all fully functional up to that boundary.
```
