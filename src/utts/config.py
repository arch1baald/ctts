import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# path to the .env file
ENV_PATH: Path = Path(os.getenv("UTTS_ENV", ".env"))

# 0 means no caching, None means unlimited lru_cache
MAXHITS: int | None = 0 if os.getenv("UTTS_CACHE_MAXHITS") != "null" else None

# request timeout (default: 10 seconds)
TIMEOUT: int = int(os.getenv("UTTS_TIMEOUT", 10))


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

    # TODO: that was probably a mistake. Should have a UTTSClient object instead.

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


def get_settings() -> Settings:
    """Returns an instance of the application settings."""
    env_path = Path(ENV_PATH)

    if not env_path.exists():
        raise FileNotFoundError(f"The file does not exist: {env_path}")

    settings_kwargs: dict[str, Any] = {"_env_file": str(env_path)}
    return Settings(**settings_kwargs)
