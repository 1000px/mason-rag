from typing import Dict, List

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import get_llm_config
from src.core.vectorstore import VectorStore

RAG_SYSTEM_PROMPT = """你是一个企业知识库智能助手。请严格基于以下检索到的企业资料来回答问题。

规则：
1. 如果资料中有明确答案，直接引用并标注来源
2. 如果资料中只有部分相关信息，说明"根据现有资料，以下是相关内容：..."
3. 如果资料中完全没有相关信息，明确回答"抱歉，知识库中暂无相关信息"
4. 禁止编造或猜测资料中不存在的内容
5. 回答结构清晰，使用中文

检索到的资料：
{context}"""


class RAGEngine:
    def __init__(self, collection_name: str = "mason_rag_docs"):
        llm_cfg = get_llm_config()

        self.llm = ChatOpenAI(
            model=llm_cfg["model_name"],
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
            temperature=0.3,
        )

        self.vector_store = VectorStore(collection_name=collection_name)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(
            self.vector_store.store.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain,
        )

        self._sql_engine = None

    @property
    def sql_engine(self):
        if self._sql_engine is None:
            from src.core.sql_engine import SqlQueryEngine
            self._sql_engine = SqlQueryEngine()
        return self._sql_engine

    def query(self, question: str) -> Dict:
        sql_result = None
        if self.sql_engine.can_handle(question):
            sql_result = self.sql_engine.query(question)

        if sql_result and sql_result.get("success"):
            answer = self.sql_engine.generate_answer(question, sql_result)
            sources = ["[结构化数据查询]"]
            return {
                "answer": answer,
                "sources": sources,
                "mode": "sql",
                "sql_data": {
                    "columns": sql_result.get("columns", []),
                    "rows": sql_result.get("rows", []),
                    "row_count": sql_result.get("row_count", 0),
                },
            }

        result = self.chain.invoke({"input": question})

        sources = []
        seen = set()
        for doc in result.get("context", []):
            file_name = doc.metadata.get("file_name", "未知")
            if file_name not in seen:
                sources.append(file_name)
                seen.add(file_name)

        answer = result["answer"]

        if sql_result and not sql_result.get("success"):
            answer = f"（尝试了数据查询但失败: {sql_result.get('error', '')}）\n\n回退到文档检索结果：\n\n{answer}"

        return {
            "answer": answer,
            "sources": sources,
            "mode": "rag",
        }

    def search_documents(self, question: str, top_k: int = 5) -> List[Document]:
        return self.vector_store.search(question, top_k=top_k)

    def get_table_info(self) -> Dict:
        from src.core.table_store import TableStore
        ts = TableStore()
        return {
            "tables": ts.get_table_names(),
            "files": ts.list_files(),
            "schema": ts.get_schema_for_llm(),
        }