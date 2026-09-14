"""
AI Avatar Video Generator - Application Settings

Uses pydantic-settings for environment variable management.
Load configuration from .env file or environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AvatarSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Dedicated env file so avatar config stays isolated from other
        # projects that share this workspace root (their .env uses different
        # values, e.g. a Docker-internal REDIS_URL that would break local runs).
        env_file=".env.avatar",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Output directory for generated videos and task artifacts
    avatar_output_dir: Path = Path(
        r"C:\Users\ariel\OneDrive\Desktop\ai-avatar-output"
    )

    # Redis connection URL (Celery broker + task state storage)
    redis_url: str = "redis://localhost:6379/0"

    # Directory containing AI model weights
    model_dir: Path = Path(
        r"C:\Users\ariel\OneDrive\Desktop\ai-avatar-output\models"
    )

    @property
    def tasks_dir(self) -> Path:
        """Directory for storing task-specific files."""
        return self.avatar_output_dir / "tasks"

    @property
    def models_dir(self) -> Path:
        """Alias for model_dir for clarity."""
        return self.model_dir

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.avatar_output_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = AvatarSettings()
