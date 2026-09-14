"""AI Avatar Video Generator - Core data models and enums.

All Pydantic models defining the system's data structures for task management,
processing results, validation, and WebSocket notifications.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class TaskStage(str, Enum):
    """處理管線的各階段名稱"""

    VOICE_ANALYSIS = "聲紋分析"
    VOICE_SYNTHESIS = "語音合成"
    FACE_MODELING = "人臉建模"
    LIP_SYNC = "唇形同步"
    VIDEO_COMPOSE = "影片合成"


class TaskState(str, Enum):
    """任務狀態"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class LanguageType(str, Enum):
    """語言類型標記"""

    CHINESE = "zh"
    ENGLISH = "en"
    UNKNOWN = "unknown"


class LanguageSegment(BaseModel):
    """講稿中的語言區段"""

    text: str
    language: LanguageType
    start_index: int
    end_index: int


class ScriptResult(BaseModel):
    """講稿處理結果"""

    segments: list[LanguageSegment]
    paragraph_breaks: list[int]  # 段落分隔在字元流中的位置
    total_chars: int


class VoiceProfile(BaseModel):
    """聲音樣本分析結果"""

    sample_path: str
    duration_sec: float
    sample_rate: int
    embedding: list[float]  # 聲紋特徵向量


class FaceInfo(BaseModel):
    """人臉偵測資訊"""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    confidence: float
    embedding: list[float]  # 人臉特徵向量
    landmarks: list[tuple[float, float]]  # 臉部關鍵點


class SynthesisResult(BaseModel):
    """語音合成結果"""

    audio_path: str
    duration_sec: float
    sample_rate: int
    bit_depth: int


class LipSyncResult(BaseModel):
    """唇形同步結果"""

    video_path: str
    fps: float
    duration_sec: float
    resolution: tuple[int, int]


class ComposeResult(BaseModel):
    """影片合成結果"""

    video_path: str
    audio_mp3_path: str
    duration_sec: float
    resolution: tuple[int, int]
    fps: float
    av_sync_offset_ms: float


class ValidationResult(BaseModel):
    """輸入驗證結果"""

    is_valid: bool
    errors: list[str]


class TaskProgress(BaseModel):
    """任務進度狀態"""

    task_id: str
    state: TaskState
    current_stage: TaskStage | None
    percentage: int  # 0-100
    error_message: str | None
    error_stage: TaskStage | None


class Task(BaseModel):
    """完整任務資料"""

    task_id: str
    state: TaskState
    created_at: datetime
    updated_at: datetime
    progress: TaskProgress
    voice_sample_path: str
    appearance_asset_path: str
    script_text: str
    output_video_path: str | None
    output_audio_path: str | None
    download_expires_at: datetime | None  # 建立後 24 小時


class TaskInput(BaseModel):
    """任務建立輸入"""

    voice_sample: bytes
    voice_sample_filename: str
    appearance_asset: bytes
    appearance_asset_filename: str
    script_text: str


class ErrorNotification(BaseModel):
    """WebSocket 錯誤通知格式"""

    type: str = "error"
    task_id: str
    stage: TaskStage  # 失敗的階段名稱
    error_category: str  # 錯誤原因類別
    message: str  # 使用者可讀的錯誤訊息
    recoverable: bool  # 是否可重試
    retry_from_stage: TaskStage | None  # 重試起始階段
