from functools import lru_cache
from pathlib import Path

from typing import Any

from pydantic import AliasChoices, Field, PositiveInt, field_validator
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
    vocab_size: PositiveInt
    context_length: PositiveInt = Field(
        validation_alias=AliasChoices("CONTEXT_LENGTH", "MAX_LENGTH")
    )
    emb_dim: PositiveInt = Field(
        validation_alias=AliasChoices("EMB_DIM", "EMBEDDING_DIM")
    )
    num_heads: PositiveInt = Field(
        validation_alias=AliasChoices("NUM_HEADS", "N_HEADS")
    )
    num_layers: PositiveInt = Field(
        validation_alias=AliasChoices("NUM_LAYERS", "N_LAYERS")
    )
    drop_rate: float
    dropout: float
    qkv_bias: bool

    @field_validator("text_path")
    @classmethod
    def resolve_text_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()

@lru_cache
def get_settings() -> Settings:
    return Settings()
