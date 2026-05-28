from __future__ import annotations

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "gigachat"
    # порог по качеству ответа
    quality_threshold: float = 0.75
    max_model_retries: int = 1
    # кол-во документов из кб, которые отдаются в промпт, при поиске по базе знаний
    kb_top_k: int = 3
    
    top_p: float = 0.9
    timeout_seconds: int = 90
    log_level: str = "INFO"

    gigachat_credentials: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GIGACHAT_API_KEY", "GIGACHAT_CREDENTIALS"),
    )
    gigachat_scope: str = "GIGACHAT_API_CORP"
    gigachat_model: str = "GigaChat-2-Max"
    # взял эмбеддинги попроще, т.к. датасет простой 
    embeddings_model: str = "Embeddings-2"

    # веса для гибридного поиска по базе знаний, поставил больший вес для BM25,
    #  т.к. эмбеддинги не всегда корректно обрабатывают технические термины и 
    # могут отдавать нерелевантные результаты, в то время как BM25 хорошо справляется с такими случаями.
    # для демо это лучше подходит
    retriever_weight_chroma: float = Field(default=0.5, validation_alias="RETRIEVER_WEIGHT_CHROMA")
    retriever_weight_bm25: float = Field(default=0.5, validation_alias="RETRIEVER_WEIGHT_BM25")

    dataset_path: str = Field(default="dataset", validation_alias="DATASET_PATH")

    # логирование в LangSmith для визуализации цепочек в интерфейсе LangSmith, 
    # можно отключить, если не нужно.
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = "Smart_support_agent"
    langsmith_endpoint: str | None = Field(default="https://api.smith.langchain.com", validation_alias="LANGSMITH_ENDPOINT")

    @property
    def retriever_weights(self) -> dict[str, float]:
        return {
            "chroma": self.retriever_weight_chroma,
            "bm25": self.retriever_weight_bm25,
        }

    @staticmethod
    def from_env() -> Settings:
        return Settings()

def configure_langsmith(settings: Settings) -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
