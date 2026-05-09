from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    llm_backend: Literal["heuristic", "openai"] = "heuristic"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    database_url: str = Field(
        default="mysql+pymysql://appuser:apppass@127.0.0.1:3306/customer_service"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

