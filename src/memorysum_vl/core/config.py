from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Track(StrEnum):
    TEXT = "text"
    CODE = "code"
    MULTIMODAL = "multimodal"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORYSUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "memorysum-vl"
    app_version: str = "0.1.0"
    api_prefix: str = "/v1"
    track: Track = Track.MULTIMODAL
    log_level: str = "INFO"
    auth_token: SecretStr | None = None
    auth_token_file: Path | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
