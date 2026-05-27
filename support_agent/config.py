from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    quality_threshold: float = 0.75
    max_refinements: int = 2
    kb_top_k: int = 3
    top_p: float = 0.9
    timeout_seconds: int = 90
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
        load_dotenv()
        return Settings(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.75")),
            max_refinements=int(os.getenv("MAX_REFINEMENTS", "2")),
            kb_top_k=int(os.getenv("KB_TOP_K", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            gigachat_credentials=os.getenv("GIGACHAT_API_KEY")
            or os.getenv("GIGACHAT_CREDENTIALS"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "Embeddings-2").strip(),
            gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP"),
            gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
            top_p=float(os.getenv("TOP_P", "0.9")),
            retriever_weights={
                "chroma": float(os.getenv("RETRIEVER_WEIGHT_CHROMA", "0.5")),
                "bm25": float(os.getenv("RETRIEVER_WEIGHT_BM25", "0.5")),},
            
            files=_parse_files(os.getenv("FILES")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "150")),
        )


def _load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_files(value: str | None) -> list[str]:
    if not value:
        return []

    stripped = value.strip().strip("[]")
    return [item.strip().strip("\"'") for item in stripped.split(",") if item.strip()]
            
