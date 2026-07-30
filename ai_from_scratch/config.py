from functools import lru_cache
from pathlib import Path

from pydantic import Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    text_path: Path = Field(default=PROJECT_ROOT / "the-verdict.txt")

    @field_validator("text_path")
    @classmethod
    def resolve_text_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

@lru_cache
def get_settings() -> Settings:
    return Settings()
