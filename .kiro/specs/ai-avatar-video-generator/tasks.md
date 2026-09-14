# Implementation Plan: AI Avatar Video Generator

## Overview

本實作計畫將 AI 虛擬人物影片生成系統的設計轉換為可逐步執行的程式碼任務。系統採用 FastAPI 後端、React+Vite 前端、Celery+Redis 任務佇列，以及 GPT-SoVITS、SadTalker、InsightFace、FFmpeg 等開源 AI 模組。每個任務聚焦於可編碼的實作步驟，按照依賴順序排列。

## Tasks

- [x] 1. Set up project structure, core interfaces, and data models
  - [x] 1.1 Create project directory structure and configuration files
    - Create backend directory structure: `api/routes/`, `core/`, `modules/`, `models/`, `tests/`
    - Create `pyproject.toml` or `requirements.txt` with dependencies: fastapi, uvicorn, celery, redis, pydantic, python-multipart, websockets, hypothesis, pytest
    - Create `.env.example` with `AVATAR_OUTPUT_DIR`, `REDIS_URL`, `MODEL_DIR` variables
    - Create `config/settings.py` with Pydantic BaseSettings for environment variables
    - _Requirements: All (infrastructure)_

  - [x] 1.2 Define core data models and enums
    - Implement all Pydantic models in `models/schemas.py`: TaskStage, TaskState, LanguageType, LanguageSegment, ScriptResult, VoiceProfile, FaceInfo, SynthesisResult, LipSyncResult, ComposeResult, ValidationResult, TaskProgress, Task, TaskInput
    - Implement error classes in `core/exceptions.py`: PipelineError, VoiceSynthesisError, LipSyncError, VideoComposeError, CompositionTimeoutError
    - Implement WebSocket notification model: ErrorNotification
    - _Requirements: All (data layer)_

  - [x] 1.3 Set up testing framework
    - Create `conftest.py` with shared fixtures (temp directories, sample file generators)
    - Create `pytest.ini` or `pyproject.toml` pytest config with markers for `property`, `unit`, `integration`
    - Verify Hypothesis is installed and runnable
    - _Requirements: All (testing infrastructure)_

- [x] 2. Implement input validators
  - [x] 2.1 Implement VoiceSampleValidator
    - Create `core/validators/voice_validator.py`
    - Implement format check (wav, mp3, flac), sample rate check (>= 16000 Hz), duration check (3-300 sec), file size check (<= 50MB)
    - Return ValidationResult with specific violation messages for each failed check
    - Handle corrupted/undecodable files gracefully
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

  - [x] 2.2 Write property test for VoiceSampleValidator
    - **Property 1: Voice sample validation correctness**
    - Generate random (format, sample_rate, duration, file_size) combinations using Hypothesis strategies
    - Assert validator accepts if and only if all constraints are satisfied
    - Assert rejection messages list specific violations
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.5**

  - [x] 2.3 Implement AppearanceAssetValidator for images
    - Create `core/validators/appearance_validator.py`
    - Implement image format check (jpg, jpeg, png, webp), resolution check (>= 512x512), file size check (<= 20MB)
    - Return ValidationResult with specific violation messages
    - _Requirements: 2.1, 2.5, 2.6, 2.7_

  - [x] 2.4 Write property test for image asset validation
    - **Property 2: Image asset validation correctness**
    - Generate random (format, width, height, file_size) combinations
    - Assert validator accepts if and only if all constraints are satisfied
    - **Validates: Requirements 2.1, 2.5, 2.6, 2.7**

  - [x] 2.5 Implement AppearanceAssetValidator for videos
    - Extend `core/validators/appearance_validator.py` with video validation
    - Implement video format check (mp4, mov), resolution check (>= 512x512), duration check (<= 60 sec), file size check (<= 200MB)
    - Return ValidationResult with specific violation messages
    - _Requirements: 2.2, 2.6, 2.7, 2.8_

  - [x] 2.6 Write property test for video asset validation
    - **Property 3: Video asset validation correctness**
    - Generate random (format, width, height, duration, file_size) combinations
    - Assert validator accepts if and only if all constraints are satisfied
    - **Validates: Requirements 2.2, 2.6, 2.7, 2.8**

  - [x] 2.7 Implement ScriptValidator
    - Create `core/validators/script_validator.py`
    - Implement length check (1-10000 chars), non-whitespace check (at least one non-whitespace char)
    - Return ValidationResult with current character count in error message when exceeding limit
    - _Requirements: 3.1, 3.4_

  - [x] 2.8 Write unit tests for all validators
    - Test boundary conditions: audio exactly 3s and 300s, image exactly 512x512, script exactly 1 char and 10000 chars
    - Test pure whitespace script rejection
    - Test corrupted file handling
    - _Requirements: 1.2, 1.3, 2.5, 3.1, 3.4_

- [x] 3. Checkpoint - Ensure all validator tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Script Processor module
  - [x] 4.1 Implement ScriptProcessor with language segmentation
    - Create `modules/script_processor.py`
    - Implement `detect_language_segments()`: identify consecutive character runs as Chinese (CJK Unicode ranges), English (Latin/ASCII), or unknown
    - Implement `detect_paragraph_breaks()`: find positions of consecutive newline characters (2+ newlines)
    - Implement `process()`: combine segmentation and paragraph detection into ScriptResult
    - Ensure processing completes within 3 seconds for max-length scripts
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6_

  - [x] 4.2 Write property test for ScriptProcessor
    - **Property 5: Script language segmentation round-trip and correctness**
    - Generate random mixed-language text (Chinese, English, special chars, newlines) using Hypothesis
    - Assert concatenation of all segment texts reproduces original text exactly
    - Assert Chinese segments contain only CJK chars, English segments only Latin/ASCII, unknown for others
    - Assert paragraph breaks are correctly identified at consecutive double-newline positions
    - Assert rejection for empty, all-whitespace, or >10000 char inputs
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

  - [x] 4.3 Write unit tests for ScriptProcessor edge cases
    - Test single character input
    - Test text at exactly 10000 character limit
    - Test text with mixed CJK, Latin, emoji, and special symbols
    - Test paragraph break detection with various newline patterns
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [x] 5. Implement Face Detector module
  - [x] 5.1 Implement FaceDetector with InsightFace
    - Create `modules/face_detector.py`
    - Implement `__init__()` to load InsightFace model (buffalo_l)
    - Implement `detect_faces()` to return list of FaceInfo objects from image
    - Implement `select_primary_face()` to return face with largest area
    - Implement `extract_embedding()` for face similarity comparison
    - Handle no-face-detected case with appropriate error
    - _Requirements: 2.3, 2.4, 5.2_

  - [x] 5.2 Write property test for primary face selection
    - **Property 4: Primary face selection by maximum area**
    - Generate random lists of FaceInfo objects with distinct areas
    - Assert select_primary_face always returns the one with largest area
    - **Validates: Requirements 2.3**

  - [x] 5.3 Write unit tests for FaceDetector
    - Test empty face list raises appropriate error
    - Test single face returns that face
    - Test face detection error handling when model unavailable
    - _Requirements: 2.3, 2.4_

- [x] 6. Implement Voice Cloner module
  - [x] 6.1 Implement VoiceCloner with GPT-SoVITS
    - Create `modules/voice_cloner.py`
    - Implement `__init__()` to load GPT-SoVITS model on specified device (cuda/cpu)
    - Implement `analyze_voice_sample()` to extract voice profile (embedding, duration, sample rate)
    - Implement `synthesize()` to generate WAV audio (44100Hz, 16-bit) from ScriptResult and VoiceProfile
    - Insert 0.5-1.5 second silence at paragraph breaks
    - Ensure no incomplete audio file is written on failure
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 6.2 Write unit tests for VoiceCloner
    - Test synthesis failure does not produce output file
    - Test paragraph silence insertion logic
    - Test voice sample rejection for insufficient duration
    - _Requirements: 4.4, 4.6, 4.7_

- [x] 7. Implement Lip Sync Engine module
  - [x] 7.1 Implement LipSyncEngine with SadTalker
    - Create `modules/lip_sync_engine.py`
    - Implement `__init__()` to load SadTalker checkpoint weights
    - Implement `generate()` to produce lip-synced video at specified fps (>= 24 fps)
    - Configure micro-expressions: blink rate 15-20/min, head sway ±2 degrees
    - Implement idle pose generation for silences > 500ms (breathing animation + random blinks)
    - Stop and report error if audio or face data is unavailable
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 7.2 Write unit tests for LipSyncEngine
    - Test error handling when audio path is invalid
    - Test error handling when source image is invalid
    - Test fps parameter is respected in output metadata
    - _Requirements: 5.1, 5.6, 5.7_

- [x] 8. Implement Video Composer module
  - [x] 8.1 Implement VideoComposer with FFmpeg
    - Create `modules/video_composer.py`
    - Implement `compose()` to merge video and audio into MP4 (1280x720, 25fps, libx264/aac)
    - Implement `extract_audio_mp3()` to convert WAV to MP3 (128kbps)
    - Implement `get_av_sync_offset()` to measure audio-video time offset
    - Implement timeout logic: abort if elapsed > video_duration * 3 AND elapsed > 180s
    - Preserve intermediate artifacts on composition failure
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8_

  - [x] 8.2 Write property test for composition timeout decision
    - **Property 6: Composition timeout decision correctness**
    - Generate random (video_duration_sec, elapsed_time_sec) pairs
    - Assert timeout returns true iff elapsed > duration * 3 AND elapsed > 180
    - **Validates: Requirements 6.8**

  - [x] 8.3 Write unit tests for VideoComposer
    - Test MP3 extraction produces valid output
    - Test compose failure preserves intermediate files
    - Test timeout detection at boundary conditions
    - _Requirements: 6.4, 6.7, 6.8_

- [x] 9. Checkpoint - Ensure all module tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Task Manager and progress tracking
  - [x] 10.1 Implement TaskManager with Redis state storage
    - Create `core/task_manager.py`
    - Implement `create_task()` to create task directory structure and store metadata in Redis
    - Implement `get_task_status()` to retrieve TaskProgress from Redis
    - Implement `update_progress()` to update stage and percentage, enforcing strict monotonicity
    - Implement `cleanup_expired_tasks()` to remove tasks older than 24 hours
    - Set download_expires_at to 24 hours after task completion
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 10.2 Write property test for progress stage monotonicity
    - **Property 7: Progress stage percentage monotonicity**
    - Generate random stage transition sequences following defined order
    - Assert percentage at each stage start is strictly greater than previous stage start
    - **Validates: Requirements 7.2**

  - [x] 10.3 Write property test for download link temporal validity
    - **Property 8: Download link temporal validity**
    - Generate random (completion_time, access_time) timestamp pairs
    - Assert link valid iff access_time - completion_time < 86400 seconds
    - **Validates: Requirements 7.4**

  - [x] 10.4 Write unit tests for TaskManager
    - Test task creation writes correct Redis keys
    - Test progress update rejects non-monotonic percentages
    - Test cleanup removes expired tasks and preserves active ones
    - Test task state transitions (pending → processing → completed/failed)
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

- [x] 11. Implement submission validator and task input combination check
  - [x] 11.1 Implement SubmissionValidator
    - Create `core/validators/submission_validator.py`
    - Implement combined validation: check all three inputs (voice_sample, appearance_asset, script) are present and non-zero size
    - Run individual validators (voice duration 3-300s, script length 1-10000 chars)
    - List all missing/invalid items in error response
    - On success, produce summary: voice sample duration, appearance asset type, script char count
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 11.2 Write property test for task submission validation
    - **Property 9: Task submission validation completeness**
    - Generate random input combinations with K items (0-3) missing or zero-size
    - Assert rejection when K > 0, with exactly K items listed in error
    - Assert acceptance when K = 0 and all individual validations pass, with correct summary values
    - **Validates: Requirements 8.1, 8.2, 8.5**

  - [x] 11.3 Write unit tests for SubmissionValidator
    - Test each individual input missing/zero-size
    - Test all inputs valid produces correct summary
    - Test voice duration out of range blocks submission
    - Test script length out of range blocks submission
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 12. Checkpoint - Ensure all core logic tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement FastAPI routes and WebSocket handler
  - [x] 13.1 Implement REST API routes
    - Create `api/routes/tasks.py`
    - Implement `POST /api/v1/tasks`: accept multipart form (voice_sample, appearance_asset, script_text), run validators, create Celery task, return task_id
    - Implement `GET /api/v1/tasks/{task_id}`: return TaskProgress from Redis
    - Implement `GET /api/v1/tasks/{task_id}/download/video`: serve final MP4 file with expiry check
    - Implement `GET /api/v1/tasks/{task_id}/download/audio`: serve final MP3 file with expiry check
    - Implement `DELETE /api/v1/tasks/{task_id}`: delete task directory and Redis keys
    - Handle upload timeout (120s no data) with 408 response and temp file cleanup
    - _Requirements: 1.4, 1.6, 6.5, 7.4, 8.1_

  - [x] 13.2 Implement WebSocket progress handler
    - Create `api/routes/ws.py`
    - Implement `WS /api/v1/ws/tasks/{task_id}/progress`: subscribe to Redis pub/sub for task updates
    - Push progress updates (stage name + percentage) at least every 3 seconds
    - Push ErrorNotification on failure with stage, error category, and retry info
    - Handle client reconnection: send current state on new connection
    - _Requirements: 7.1, 7.3, 7.5_

  - [x] 13.3 Create FastAPI application entry point
    - Create `main.py` with FastAPI app, include routers, configure CORS for frontend
    - Add startup/shutdown events for Redis connection pool
    - Configure upload size limits matching validator constraints
    - _Requirements: All (application wiring)_

  - [x] 13.4 Write unit tests for API routes
    - Test POST /tasks with valid inputs returns task_id
    - Test POST /tasks with missing inputs returns 400 with item list
    - Test GET /tasks/{id} returns current progress
    - Test download endpoints return 404 for expired links
    - Test upload timeout handling
    - _Requirements: 1.4, 1.6, 7.4, 8.1, 8.2_

- [x] 14. Implement Celery worker and processing pipeline
  - [x] 14.1 Implement Celery task pipeline
    - Create `core/celery_app.py` with Celery configuration (Redis broker/backend)
    - Create `core/pipeline.py` with main generation task
    - Implement pipeline stages in order: script processing → voice synthesis → face modeling → lip sync → video composition
    - Update progress via TaskManager at each stage transition
    - Handle PipelineError at each stage: stop processing, mark task failed, send error notification
    - Implement retry-from-stage capability: skip completed stages on retry
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 14.2 Write unit tests for pipeline error handling
    - Test pipeline stops on VoiceSynthesisError (no incomplete audio written)
    - Test pipeline stops on LipSyncError
    - Test pipeline preserves intermediate files on VideoComposeError
    - Test retry starts from correct failed stage
    - Test progress updates are monotonic through pipeline
    - _Requirements: 4.7, 5.6, 6.7, 7.3_

- [x] 15. Checkpoint - Ensure all backend tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement React frontend
  - [x] 16.1 Set up React + Vite frontend project
    - Create `frontend/` directory with Vite React TypeScript template
    - Install dependencies: react, react-dom, axios (or fetch), tailwindcss or similar CSS framework
    - Configure proxy to FastAPI backend for development
    - _Requirements: All (frontend infrastructure)_

  - [x] 16.2 Implement file upload components
    - Create VoiceUpload component: accept WAV/MP3/FLAC, max 50MB, show validation errors
    - Create AppearanceUpload component: accept JPG/PNG/WEBP/MP4/MOV, show validation errors
    - Create ScriptInput component: textarea with character counter (max 10000), preserve content on error
    - Show upload progress indicator within 2 seconds of successful upload
    - _Requirements: 1.1, 1.4, 1.5, 2.1, 2.2, 2.6, 3.1, 3.4_

  - [x] 16.3 Implement task submission and summary display
    - Create SubmissionForm component: combine all inputs, show material summary before confirm
    - Display voice sample duration, appearance asset type, script character count
    - Require user confirmation button before starting generation
    - Show per-item validation errors preventing submission
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 16.4 Implement progress tracking and WebSocket connection
    - Create ProgressDisplay component: show current stage name and percentage (0-100%)
    - Connect to WebSocket endpoint for real-time updates
    - Handle reconnection: restore current progress state on page return
    - Display error messages with failed stage name, error category, and retry button
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

  - [x] 16.5 Implement video preview and download
    - Create VideoPreview component: embed video player for playback before download
    - Create download buttons for video (MP4) and audio (MP3) separately
    - Show download link expiry information (24-hour validity)
    - Handle expired link gracefully with appropriate message
    - _Requirements: 6.4, 6.5, 7.4_

- [x] 17. Integration wiring and end-to-end flow
  - [x] 17.1 Wire frontend to backend API
    - Connect upload components to `POST /api/v1/tasks` endpoint
    - Connect progress display to WebSocket endpoint
    - Connect download buttons to download endpoints
    - Connect retry button to task re-submission with retry_from_stage parameter
    - _Requirements: All (integration)_

  - [x] 17.2 Write integration tests for end-to-end pipeline
    - Test full flow: upload → validate → process → download (requires GPU)
    - Test WebSocket progress updates arrive within 3 seconds
    - Test retry from failed stage resumes correctly
    - Test expired download link returns appropriate error
    - _Requirements: 7.1, 7.3, 7.4, 7.5_

- [x] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property-based tests use Hypothesis framework with minimum 100 iterations per property
- Integration tests (17.2) require NVIDIA GPU with 8GB+ VRAM for AI model inference
- All AI inference runs locally — no cloud API keys needed
- Checkpoints ensure incremental validation at logical boundaries
- The frontend (task 16) can be developed in parallel with backend tasks 10-14 after core modules are complete
- Default output directory: `C:\Users\ariel\OneDrive\Desktop\ai-avatar-output\` (configurable via `AVATAR_OUTPUT_DIR` env var)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.3", "2.5", "2.7"] },
    { "id": 2, "tasks": ["2.2", "2.4", "2.6", "2.8", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "6.1"] },
    { "id": 5, "tasks": ["6.2", "7.1", "8.1"] },
    { "id": 6, "tasks": ["7.2", "8.2", "8.3", "10.1", "11.1"] },
    { "id": 7, "tasks": ["10.2", "10.3", "10.4", "11.2", "11.3"] },
    { "id": 8, "tasks": ["13.1", "13.2", "13.3", "16.1"] },
    { "id": 9, "tasks": ["13.4", "14.1", "16.2", "16.3"] },
    { "id": 10, "tasks": ["14.2", "16.4", "16.5"] },
    { "id": 11, "tasks": ["17.1"] },
    { "id": 12, "tasks": ["17.2"] }
  ]
}
```
