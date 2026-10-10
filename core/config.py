from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent

ENV_TEMPLATE_FILE = PROJECT_DIR / ".env.template"
ENV_FILE = PROJECT_DIR / ".env"

MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "data"


class LoggerSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = (
        "[%(asctime)s.%(msecs)03d] %(module)10s:%(lineno)-3d %(levelname)-7s - %(message)s"
    )
    datefmt: str = "%Y-%m-%d %H:%M:%S"



class BotSettings(BaseModel):
    token: SecretStr
    developers: list[str] = Field(default_factory=list, repr=False)
    help_text: str = (
        'Отправь текстовый вопрос по ККИ "Берсерк", и я поищу ответ в базе знаний.\n\n'
        'Например: "Как работает атака?" или "Что такое Провокация?"\n\n'
        "Каждый вопрос рассматривается отдельно, без истории переписки.\n"
        "/start - начать\n"
        "/help - помощь"
    )

    about_game_text: str = (
        '"Берсерк" - коллекционная карточная игра в жанре фэнтези. '
        "Игроки собирают отряды из карт существ и сражаются, используя их "
        "способности и продумывая тактику. Цель - победить отряд соперника.\n\n"
        "Бот помогает разобраться в правилах, картах и игровых механиках: "
        "задай вопрос, и он поищет подходящую информацию в базе знаний."
    )

    about_bot_text: str = (
        'Берсерк AI - помощник по ККИ "Берсерк". '
        "Он ищет информацию в базе знаний и с помощью языковой модели "
        "формирует ответ на твой вопрос.\n\n"
        "Каждый вопрос рассматривается отдельно, без истории переписки. "
        "Ответы могут быть неточными - спорные ситуации сверяй с правилами игры."
    )


    settings_stub_text: str = "Настройки пока в разработке. Этот раздел появится позже."


class QdrantSettings(BaseModel):
    url: str
    api_key: SecretStr
    collections: list[str] = ["cards", "rules", "faq", "community"]


class ModelsSettings(BaseModel):
    dense: str = "intfloat/multilingual-e5-large"
    sparse: str = "Qdrant/bm25"
    rerank: str = "jinaai/jina-reranker-v2-base-multilingual"


class RetrievalSettings(BaseModel):
    top_k: int = 5
    dense_limit: int = 20
    sparse_limit: int = 20
    rrf_limit: int = 20
    use_reranker: bool = True


class GuardSettings(BaseModel):
    # score < refuse_threshold -> отказ без вызова LLM
    # refuse_threshold <= score < warn -> ответ с предупреждением
    warn_threshold: float = 0.5
    refuse_threshold: float = 0.1

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.refuse_threshold > self.warn_threshold:
            raise ValueError("guard.refuse_threshold должен быть <= guard.warn_threshold")
        return self


class LLMSettings(BaseModel):
    provider: Literal["ollama", "openai"]
    base_url: str
    model: str
    api_key: SecretStr
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: float = 60.0
    max_concurrency: int = Field(default=1, ge=1)  # 1 - локальный GPU, генерация по очереди

    system_prompt: str = """Ты - полезный AI-ассистент.
Отвечай только на основе предоставленного контекста.
Если ответа нет в контексте, скажи "Я не знаю".
Не выдумывай факты."""

    @model_validator(mode="after")
    def _check_api_key(self) -> Self:
        if self.provider == "openai" and not self.api_key.get_secret_value():
            raise ValueError(
                "llm.api_key не может быть пустым для provider=openai "
                "(для локального OpenAI-совместимого сервера подойдёт любая строка)"
            )
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[ENV_TEMPLATE_FILE, ENV_FILE],
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    bot: BotSettings
    qdrant: QdrantSettings
    llm: LLMSettings

    log: LoggerSettings = LoggerSettings()
    models: ModelsSettings = ModelsSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    guard: GuardSettings = GuardSettings()


def ensure_directories() -> None:
    for directory in (MODELS_DIR, LOGS_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    ensure_directories()
    return Settings()  # type: ignore[call-arg]  # обязательные поля приходят из .env
