from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_gigachat import GigaChatEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from support_agent.config import Settings


class KnowledgeBase:
    def search(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        top_k: int = 3,
    ) -> list[Document]:
        raise NotImplementedError


class HybridChromaKnowledgeBase(KnowledgeBase):
    def __init__(
        self,
        documents: list[Document],
        settings: Settings,
        collection_name: str = "support_kb_hybrid",
        chunk_size: int = 700,
        chunk_overlap: int = 80,
        chroma_batch_size: int = 4,
    ):
        if not settings.gigachat_credentials:
            raise RuntimeError("GIGACHAT_CREDENTIALS is required for GigaChatEmbeddings")

        self.source_documents = documents
        # При подключении БД были большие документы, поэтому порезал на чанки, т.к. эмбеддинги ограничены по размеру текста. 
        self.documents = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.settings = settings
        self.embeddings = GigaChatEmbeddings(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            model=settings.embeddings_model,
            verify_ssl_certs=False,
            timeout=60,
        )
        self.bm25 = BM25Retriever.from_documents(self.documents)
        self.chroma = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        for start in range(0, len(self.documents), chroma_batch_size):
            self.chroma.add_documents(self.documents[start : start + chroma_batch_size])

    # пример метода для построения базы знаний из тестовых Markdown файлов с вопросами и ответами.
    @classmethod
    def from_markdown_files(
        cls,
        settings: Settings,
        collection_name: str = "from_markdown",
        chunk_size: int = 700,
        chunk_overlap: int = 80,
        chroma_batch_size: int = 4,
    ) -> HybridChromaKnowledgeBase:
        docs: list[Document] = []
        for path in Path(settings.dataset_path).glob("*.md"):
            docs.extend(build_documents_from_markdown(path))
        return cls(
            documents=docs,
            settings=settings,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chroma_batch_size=chroma_batch_size,
        )

    # пример метода для построения базы знаний из PostgreSQL базы данных.
    @classmethod
    def from_postgres(
        cls,
        settings: Settings,
        collection_name: str = "from_postgres",
        chunk_size: int = 700,
        chunk_overlap: int = 80,
        chroma_batch_size: int = 4,
    ) -> "HybridChromaKnowledgeBase":
        """Example method to build a knowledge base from a PostgreSQL database. """
        url = URL.create(
            "postgresql+psycopg",
            username="postgres",
            password="postgres",
            host="localhost",
            port=5432,
            database="postgres",
        )
        # Взял свою тестовую таблицу, которая не имеет отношения к тикетам поддержки, просто для демонстрации
        query = "SELECT id, title, content, description FROM extracted_data"

        engine = create_engine(url)
        try:
            df = pd.read_sql_query(query, engine)
        finally:
            engine.dispose()

        documents: list[Document] = []
        for _, row in df.loc[:25, ["id", "title", "content", "description"]].iterrows():
            content = str(row["content"])
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": row["id"],
                        "title": row["title"],
                        "content": row["content"],
                        "description": row["description"],
                    },
                )
            )
        return cls(
            documents=documents,
            settings=settings,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chroma_batch_size=chroma_batch_size,
        )

    def search(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        top_k: int = 3,
    ) -> list[Document]:
        """Search the knowledge base using a hybrid of BM25 and dense retrieval."""
        search_kwargs = {"k": max(top_k, 5)}

        if filters and filters.get("source") is not None:
            search_kwargs["filter"] = {"source": {"$eq": filters["source"]}}


        dense = self.chroma.as_retriever(search_type="similarity", 
                                         search_kwargs=search_kwargs)
        # В реальной реализации можно динамически регулировать веса в зависимости от запроса или других факторов, но для простоты сейчас они фиксированные.
        # Я поставил в . env больший вес для sparse исходя из тестовых данных
        hybrid = EnsembleRetriever( 
            retrievers=[dense, self.bm25],
            weights=[self.settings.retriever_weights.get("chroma") or 0.5, self.settings.retriever_weights.get("bm25") or 0.5],
        )

        results = hybrid.invoke(query)
        filtered = apply_filters(results, filters or {})
        trimmed = filtered[:top_k]
        return trimmed
        


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 80,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        base_metadata = {
            key: value
            for key, value in doc.metadata.items()
            if key not in {"answer", "content", "description"}
        }
        split_texts = splitter.split_text(doc.page_content)
        chunk_count = len(split_texts)

        for index, text in enumerate(split_texts):
            metadata = {
                **base_metadata,
                "answer": text,
                "chunk_index": index,
                "chunk_count": chunk_count,
            }
            if "id" in base_metadata:
                metadata["parent_id"] = base_metadata["id"]

            chunks.append(Document(page_content=text, metadata=metadata))

    return chunks


def build_documents_from_markdown(path: Path) -> list[Document]:
    entries = parse_markdown_qa(path)
    docs: list[Document] = []
    for entry in entries:
        docs.append(
            Document(
                page_content=
                        f"{entry['answer']}",
                metadata={
                    "id": entry["id"],
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
