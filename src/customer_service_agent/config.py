from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    llm_backend: str = "openai"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    google_model: str = "gemma-4-31b-it"
    google_api_key: str | None = None
    database_url: str = Field(
        default="mysql+pymysql://appuser:apppass@127.0.0.1:3306/customer_service"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
