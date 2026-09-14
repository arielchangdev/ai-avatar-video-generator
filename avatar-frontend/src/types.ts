// Shared domain types mirroring the backend Pydantic schemas.

export type TaskState = "pending" | "processing" | "completed" | "failed";

// Stage display names come from the backend TaskStage enum (Chinese labels).
export type TaskStage =
  | "聲紋分析"
  | "語音合成"
  | "人臉建模"
  | "唇形同步"
  | "影片合成";

export interface TaskProgress {
  task_id: string;
  state: TaskState;
  current_stage: TaskStage | null;
  percentage: number;
  error_message: string | null;
  error_stage: TaskStage | null;
}

export interface SubmissionSummary {
  voice_duration_sec: number;
  appearance_type: "image" | "video";
  script_char_count: number;
}

export interface CreateTaskResponse {
  task_id: string;
  state: TaskState;
  summary: SubmissionSummary | null;
}

// WebSocket message shapes.
export interface ProgressMessage extends TaskProgress {
  type: "progress";
}

export interface ErrorMessage {
  type: "error";
  task_id?: string;
  stage?: TaskStage;
  error_category?: string;
  message: string;
  recoverable?: boolean;
  retry_from_stage?: TaskStage | null;
}

export type WsMessage = ProgressMessage | ErrorMessage;

// Validation constraints shared with the backend validators.
export const VOICE_MAX_MB = 50;
export const VOICE_FORMATS = ["wav", "mp3", "flac"];
export const IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"];
export const VIDEO_FORMATS = ["mp4", "mov"];
export const IMAGE_MAX_MB = 20;
export const VIDEO_MAX_MB = 200;
export const SCRIPT_MAX_CHARS = 10000;
