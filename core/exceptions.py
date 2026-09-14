"""AI Avatar Video Generator - Pipeline error classes.

Custom exception hierarchy for handling errors in each processing stage
of the video generation pipeline.
"""

from models.schemas import TaskStage


class PipelineError(Exception):
    """管線處理錯誤基類"""

    def __init__(self, stage: TaskStage, reason: str, recoverable: bool = True):
        self.stage = stage
        self.reason = reason
        self.recoverable = recoverable
        super().__init__(f"[{stage.value}] {reason}")

    def __reduce__(self):
        # Make the exception picklable so Celery can serialize task failures
        # across process boundaries (rebuild from its original constructor args).
        return (self.__class__, (self.stage, self.reason, self.recoverable))


class VoiceSynthesisError(PipelineError):
    """語音合成失敗"""

    pass


class LipSyncError(PipelineError):
    """唇形同步失敗"""

    pass


class VideoComposeError(PipelineError):
    """影片合成失敗"""

    pass


class CompositionTimeoutError(PipelineError):
    """合成逾時"""

    pass
