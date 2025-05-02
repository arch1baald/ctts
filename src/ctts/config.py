from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


class OpenAISettings(BaseModel):
    """Settings for OpenAI API."""

    api_key: str = Field(..., description="OpenAI API key")
    organization_id: str | None = Field(None, description="OpenAI organization ID")


class Settings(BaseSettings):
    """Main application settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)

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
        env_path = ROOT_DIR / ".env"

    if env_path.exists():
        return Settings(_env_file=str(env_path))

    # If .env is not found, but variables may be in the environment
    return Settings()
