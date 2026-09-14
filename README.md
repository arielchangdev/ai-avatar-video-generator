# AI Avatar Video Generator

本地、零費用、零雲端依賴的 AI 虛擬人物影片生成系統。上傳一段聲音樣本、一張人物圖片（或影片）與一段講稿，系統會克隆聲音、合成語音、對嘴、並輸出一段自然的講解影片。

所有 AI 推論都在本地 GPU 上執行，不使用任何雲端付費 API。

## 技術棧

| 層 | 技術 |
|----|------|
| 語音克隆 | GPT-SoVITS |
| 唇形同步 | SadTalker |
| 人臉偵測 | InsightFace |
| 影片合成 | FFmpeg |
| 後端 | FastAPI + Celery + Redis |
| 前端 | React + Vite (TypeScript) |

## 架構

```
使用者 → React 前端 → FastAPI (REST + WebSocket)
                          │
                          ├── 輸入驗證 (聲音 / 外觀 / 講稿)
                          ├── TaskManager (Redis 狀態儲存)
                          └── Celery 佇列
                                  │
                                  ▼
        講稿處理 → 語音合成 → 人臉建模 → 唇形同步 → 影片合成
        (10%)      (40%)      (55%)      (80%)      (100%)
                                  │
                                  ▼
                           MP4 影片 + MP3 音訊
```

## 專案結構

```
api/                REST 路由 + WebSocket + 相依注入
core/               管線協調、任務管理、驗證器、Celery 設定
  validators/       聲音 / 外觀 / 講稿 / 提交 驗證器
models/schemas.py   Pydantic 資料模型
modules/            AI 模組包裝層 (script/voice/face/lipsync/compose)
avatar-frontend/    React + Vite 前端
tests/avatar/       屬性測試 + 單元測試 + 整合測試
main.py             FastAPI 進入點
```

## 快速開始

完整安裝步驟（含 GPU 與模型下載）請見
[`.kiro/specs/ai-avatar-video-generator/SETUP.md`](.kiro/specs/ai-avatar-video-generator/SETUP.md)。

### 不需要 GPU 也能跑的部分（應用骨架）

```powershell
# 1. 後端相依
python -m pip install -r requirements.txt

# 2. 環境設定
copy .env.avatar.example .env.avatar   # 再依需要修改路徑

# 3. Redis (Docker)
docker run -d --name avatar-redis -p 6379:6379 redis:7-alpine

# 4. FFmpeg
winget install --id Gyan.FFmpeg -e

# 5. 後端 API
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# 6. Celery worker (Windows 用 solo pool)
python -m celery -A core.celery_app:celery_app worker --loglevel=info --pool=solo

# 7. 前端
cd avatar-frontend
npm install
npm run dev        # http://localhost:5173
```

到這裡整個系統都能運作：上傳、驗證、建立任務、WebSocket 即時進度。實際生成影片還需要 GPU 與 AI 模型（見 SETUP.md 第 2 節）。

## 測試

```powershell
# 後端 (屬性 + 單元 + 整合，無需 GPU)
python -m pytest tests/avatar/ -q

# 前端
cd avatar-frontend
npm test
```

目前狀態：後端 202 passed / 1 skipped（GPU-only E2E），前端 24 passed。

## 授權與成本

所有元件皆為免費開源；除了本地 GPU 的電力與硬體外，無任何雲端 API 費用。
