from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR: Path = Path(__file__).parent.parent.parent.resolve()
TIMEOUT: int = 10


class OpenAISettings(BaseModel):
    """Settings for OpenAI API."""

    api_key: str = Field(..., description="OpenAI API key")
    organization_id: str | None = Field(None, description="OpenAI organization ID")


class ElevenLabsSettings(BaseModel):
    """Settings for ElevenLabs API."""

    api_key: str = Field(..., description="ElevenLabs API key")


class ReplicateSettings(BaseModel):
    """Settings for Replicate API."""

    api_key: str = Field(..., description="Replicate API key")


class ZyphraSettings(BaseModel):
    """Settings for Zyphra API."""

    api_key: str = Field(..., description="Zyphra API key")


class HumeSettings(BaseModel):
    """Settings for Hume API."""

    api_key: str = Field(..., description="Hume API key")


class CartesiaSettings(BaseModel):
    """Settings for Cartesia API."""

    api_key: str = Field(..., description="Cartesia API key")


class Settings(BaseSettings):
    """Main application settings."""

    openai: OpenAISettings | None = Field(default=None, description="OpenAI settings")
    elevenlabs: ElevenLabsSettings | None = Field(default=None, description="ElevenLabs settings")
    replicate: ReplicateSettings | None = Field(default=None, description="Replicate settings")
    zyphra: ZyphraSettings | None = Field(default=None, description="Zyphra settings")
    hume: HumeSettings | None = Field(default=None, description="Hume settings")
    cartesia: CartesiaSettings | None = Field(default=None, description="Cartesia settings")

    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_settings(env_file: str | Path | None = None) -> Settings:
    """Returns an instance of the application settings.

    Args:
        env_file: Path to the .env file. If None, the .env file from the project root will be used.
    """
    if env_file is not None:
        env_path = Path(env_file)
    else:
        env_path: Path = ROOT_DIR / ".env"

    settings_kwargs: dict[str, Any] = {}
    if env_path.exists():
        settings_kwargs["_env_file"] = str(env_path)

    return Settings(**settings_kwargs)
