from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config.settings import get_embedding_config, get_llm_config, resolve_path


def _create_embeddings() -> Embeddings:
    emb_cfg = get_embedding_config()
    llm_cfg = get_llm_config()
    provider = emb_cfg.get("provider", "openai")
    model = emb_cfg.get("model_name") or "text-embedding-3-small"

    base_url = emb_cfg.get("base_url") or llm_cfg.get("base_url") or ""
    api_key = emb_cfg.get("api_key") or llm_cfg.get("api_key") or ""

    if not api_key:
        raise ValueError("未配置 Embedding API Key，请在 .env 中设置 EMBEDDING_API_KEY 或 LLM_API_KEY")

    if provider == "qwen":
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
        )

    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


class VectorStore:
    def __init__(self, collection_name: str = "mason_rag_docs"):
        persist_dir = str(resolve_path("data/chromadb"))
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.embeddings = _create_embeddings()
        self.collection_name = collection_name

        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.store.add_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        return self.store.similarity_search(query, k=top_k, filter=filter)

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        return self.store.similarity_search_with_relevance_scores(
            query, k=top_k, filter=filter
        )

    def delete_by_file(self, file_name: str) -> int:
        collection = self.store._collection
        results = collection.get(where={"file_name": file_name})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def list_files(self) -> List[str]:
        collection = self.store._collection
        results = collection.get()
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return []
        files = set()
        for meta in metadatas:
            if meta and "file_name" in meta:
                files.add(meta["file_name"])
        return sorted(files)

    def count(self) -> int:
        return self.store._collection.count()