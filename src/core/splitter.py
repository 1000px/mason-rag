from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import get_document_config


class DocumentSplitter:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        doc_config = get_document_config()
        self.chunk_size = chunk_size or doc_config["chunk_size"]
        self.chunk_overlap = chunk_overlap or doc_config["chunk_overlap"]

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        file_type = documents[0].metadata.get("file_type", "text")

        separators = self._get_separators(file_type)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_count"] = len(chunks)

        return chunks

    def _get_separators(self, file_type: str) -> List[str]:
        if file_type == "markdown":
            return ["\n## ", "\n### ", "\n#### ", "\n---\n", "\n\n", "\n", " ", ""]
        elif file_type == "pdf":
            return ["\n\n", "\n", "。", ".", " ", ""]
        else:
            return ["\n\n", "\n", " ", ""]