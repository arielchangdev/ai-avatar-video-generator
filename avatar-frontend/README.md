# AI Avatar Video Generator — Frontend

React + Vite (TypeScript) UI for the AI Avatar Video Generator.

> Placed in `avatar-frontend/` rather than `frontend/` because the workspace
> root already contains an unrelated `frontend/` project (Red Team Automation).

## Scripts

```bash
npm install        # install dependencies
npm run dev        # start dev server on http://localhost:5173 (proxies /api to :8000)
npm run build      # typecheck + production build
npm test           # run the vitest suite
```

## Dev proxy

`vite.config.ts` proxies `/api` (REST) and `/api/v1/ws` (WebSocket) to the
FastAPI backend at `http://localhost:8000`. Start the backend with:

```bash
uvicorn main:app --reload
```

## Flow

1. Upload voice sample (WAV/MP3/FLAC), appearance asset (image or video), enter script.
2. Review the material summary and confirm to start generation.
3. Watch live progress over WebSocket; retry from the failed stage on error.
4. Preview the result and download the MP4 / MP3 (24-hour link validity).
