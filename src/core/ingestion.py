from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from src.core.loader import load_document
from src.core.splitter import DocumentSplitter
from src.core.vectorstore import VectorStore


class IngestionPipeline:
    EXCEL_EXTENSIONS = {".xlsx", ".xls"}

    def __init__(self, collection_name: str = "mason_rag_docs"):
        self.vector_store = VectorStore(collection_name=collection_name)

    def ingest(self, file_path: str | Path) -> Dict:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        result = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "document_count": 0,
            "chunk_count": 0,
            "vector_count": 0,
            "table_count": 0,
        }

        if ext in self.EXCEL_EXTENSIONS:
            from src.core.table_store import TableStore

            print(f"\n📊 正在导入表格: {file_path.name}")
            table_store = TableStore()
            table_result = table_store.ingest_excel(file_path)
            sheet_count = len(table_result["sheets"])
            total_rows = sum(s["row_count"] for s in table_result["sheets"].values())
            result["table_count"] = sheet_count
            result["table_rows"] = total_rows
            print(f"   ✓ 表格入库完成，{sheet_count} 个Sheet，共 {total_rows} 行")

        print(f"\n📄 正在加载: {file_path.name}")
        documents = load_document(file_path)
        result["document_count"] = len(documents)
        print(f"   ✓ 解析完成，共 {len(documents)} 个段落")

        print(f"✂️  正在分块...")
        splitter = DocumentSplitter()
        chunks = splitter.split(documents)
        result["chunk_count"] = len(chunks)
        print(f"   ✓ 分块完成，共 {len(chunks)} 个块")

        print(f"📥 正在存入向量库...")
        ids = self.vector_store.add_documents(chunks)
        result["vector_count"] = len(ids)
        print(f"   ✓ 存储完成，共 {len(ids)} 条向量")

        return result

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        return self.vector_store.search(query, top_k=top_k)

    def delete_file(self, file_name: str) -> int:
        from src.core.table_store import TableStore
        count = self.vector_store.delete_by_file(file_name)
        table_store = TableStore()
        table_count = table_store.delete_by_file(file_name)
        print(f"🗑️  已删除 {file_name}，移除 {count} 条向量，{table_count} 个表")
        return count

    def list_files(self) -> List[str]:
        return self.vector_store.list_files()