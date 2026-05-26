from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_gigachat import GigaChatEmbeddings
import pandas as pd

from support_agent.config import Settings


class KnowledgeBase:
    def search(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        raise NotImplementedError


class HybridChromaKnowledgeBase(KnowledgeBase):
    def __init__(
        self,
        documents: list[Document],
        settings: Settings,
        collection_name: str = "support_kb_hybrid",
    ):
        if not settings.gigachat_credentials:
            raise RuntimeError("GIGACHAT_CREDENTIALS is required for GigaChatEmbeddings")

        self.documents = documents
        self.settings = settings
        self.embeddings = GigaChatEmbeddings(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            model=settings.embeddings_model,
            verify_ssl_certs=False,
            timeout=60,
        )
        self.bm25 = BM25Retriever.from_documents(self.documents)
        self.chroma = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name=collection_name,
        )

    @classmethod
    def from_markdown_files(
        cls,
        file_paths: list[Path],
        settings: Settings,
    ) -> "HybridChromaKnowledgeBase":
        docs: list[Document] = []
        for path in file_paths:
            docs.extend(build_documents_from_markdown(path))
        return cls(documents=docs, settings=settings)

    def search(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        # Dense retriever from Chroma
        dense = self.chroma.as_retriever(search_type="similarity", search_kwargs={"k": max(top_k, 5)})
        hybrid = EnsembleRetriever(
            retrievers=[dense, self.bm25],
            weights=[self.settings.retriever_weights.get("chroma") or 0.5, self.settings.retriever_weights.get("bm25") or 0.5],
        )

        results = hybrid.invoke(query)
        filtered = apply_filters(results, filters or {})
        trimmed = filtered[:top_k]
        return [
            {
                "id": str(doc.metadata.get("id", "")),
                "title": str(doc.metadata.get("title", "")),
                # "question": str(doc.metadata.get("question", "")),
                "content": str(doc.metadata.get("answer", "")),
                "source": str(doc.metadata.get("source", "")),
            }
            for doc in trimmed
        ]


def build_documents_from_markdown(path: Path) -> list[Document]:
    entries = parse_markdown_qa(path)
    docs: list[Document] = []
    for entry in entries:
        docs.append(
            Document(
                page_content=
                        # f"Question: {entry['question']}\n" \s
                        f"Answer: {entry['answer']}",
                metadata={
                    "id": entry["id"],
                    # "title": entry["question"],
                    # "question": entry["question"],
                    "answer": entry["answer"],
                    "source": str(path),
                },
            )
        )
    return docs


def parse_markdown_qa(path: Path) -> list[dict[str, str]]:
  
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^###\s+(TICKET-Q\d+)\s*\nQuestion:\s*(.*?)\nAnswer:\s*(.*?)(?=\n###\s+TICKET-Q|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    entries: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        entries.append(
            {
                "id": match.group(1).strip(),
                # "question": match.group(2).strip(),
                "answer": match.group(3).strip(),
            }
        )
    return entries


def apply_filters(docs: list[Document], filters: dict[str, object]) -> list[Document]:
    if not filters:
        return docs
    out: list[Document] = []
    for doc in docs:
        ok = True
        for key, value in filters.items():
            if value is None:
                continue
            # ignore unknown filter keys for this kb schema
            if key not in doc.metadata:
                continue
            if str(doc.metadata.get(key)) != str(value):
                ok = False
                break
        if ok:
            out.append(doc)
    return out
