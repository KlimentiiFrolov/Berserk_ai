from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_FILE = Path(__file__).resolve()
PROJECT_DIR = CONFIG_FILE.parent.parent


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    qdrant_url: str = Field(..., validation_alias="QDRANT_URL")

    dense_model_name: str = "intfloat/multilingual-e5-large"
    sparse_model_name: str = "Qdrant/bm25"
    rerank_model_name: str = "jinaai/jina-reranker-v2-base-multilingual"

    models_cache_dir: Path = PROJECT_DIR / "models"
    logs_dir: Path = PROJECT_DIR / "logs"
    data_dir: Path = PROJECT_DIR / "data"

    default_system_prompt: str = """Ты - полезный AI-ассистент.
Отвечай только на основе предоставленного контекста.
Если ответа нет в контексте, скажи "Я не знаю".
Не выдумывай факты."""

    def ensure_directories(self) -> None:
        for directory in (
            self.models_cache_dir,
            self.logs_dir,
            self.data_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_rag_settings() -> RAGSettings:
    settings = RAGSettings() # type: ignore
    settings.ensure_directories()
    return settings