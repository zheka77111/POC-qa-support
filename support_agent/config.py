from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    quality_threshold: float = 0.75
    max_refinements: int = 2
    kb_top_k: int = 3
    log_level: str = "INFO"
    gigachat_credentials: str | None = None
    gigachat_scope: str = "GIGACHAT_API_CORP"
    gigachat_model: str = "GigaChat-2-Max"
    embeddings_model: str = "Embeddings-2"
    retriever_weights: dict[str, float] = field(
        default_factory=lambda: {"chroma": 0.5, "bm25": 0.5}
    )
    files: list[str] = field(default_factory=list)

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.75")),
            max_refinements=int(os.getenv("MAX_REFINEMENTS", "2")),
            kb_top_k=int(os.getenv("KB_TOP_K", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            gigachat_credentials=os.getenv("GIGACHAT_API_KEY"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "Embeddings-2").strip(),
            gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP"),
            gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
            retriever_weights={
                "chroma": float(os.getenv("RETRIEVER_WEIGHT_CHROMA", "0.5")),
                "bm25": float(os.getenv("RETRIEVER_WEIGHT_BM25", "0.5")),},
            
            files = [f.strip() for f in os.getenv("FILES").split(",")] if os.getenv("FILES") else []
        )
            
