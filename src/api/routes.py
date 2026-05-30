import json
import os
import re
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config.settings import (
    _init,
    get_app_config,
    get_embedding_config,
    get_llm_config,
    get_llm_providers,
    get_yaml_config,
    resolve_path,
)

router = APIRouter()

PROJECT_ROOT = resolve_path(".")

UPLOAD_DIR = resolve_path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".xlsx", ".xls", ".docx"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    mode: str = "rag"
    sql_data: Optional[dict] = None


class TableInfoResponse(BaseModel):
    tables: list[str]
    files: list[str]


class ProfileRequest(BaseModel):
    table_name: str


class ConfigUpdate(BaseModel):
    llm_provider: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model_name: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model_name: Optional[str] = None


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/info")
async def app_info():
    app_cfg = get_app_config()
    llm_cfg = get_llm_config()
    return {
        "app": app_cfg,
        "llm": {
            "provider": llm_cfg["provider"],
            "model_name": llm_cfg["model_name"],
            "base_url": llm_cfg["base_url"],
        },
    }


@router.get("/providers")
async def list_providers():
    return {
        "llm_providers": get_llm_providers(),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from src.core.rag_engine import RAGEngine

    engine = RAGEngine()
    result = engine.query(request.question)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        mode=result.get("mode", "rag"),
        sql_data=result.get("sql_data"),
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    from src.core.rag_engine import RAGEngine, RAG_SYSTEM_PROMPT

    engine = RAGEngine()

    async def event_generator():
        from src.core.data_profiler import profile_table
        from src.core.insight_engine import discover_insights
        from src.core.table_store import TableStore

        def compress_markdown(text: str) -> str:
            import re
            text = re.sub(r"\n{2,}", "\n", text)
            return text.strip()

        store = TableStore()
        table_names = store.get_table_names()

        profile_keywords = ["数据画像", "数据概况", "数据体检", "数据诊断", "数据总览", "数据洞察", "数据质量", "数据分布", "数据亮点", "发现规律", "数据规律", "业务洞察", "关联分析", "相关性分析", "帕累托分析", "趋势检测", "趋势分析", "找规律", "挖掘规律", "数据挖掘", "分析数据", "分析这张表", "分析一下数据", "分析一下这张表", "这张表怎么样", "表概况", "整体分析", "规律", "亮点", "洞察"]

        is_profile_request = any(kw in request.question for kw in profile_keywords) and table_names

        if is_profile_request:
            profiles = {}
            insights = {}
            for tn in table_names:
                try:
                    p = profile_table(store, tn)
                    profiles[tn] = p
                    if not p.get("error"):
                        insights[tn] = discover_insights(
                            store, tn,
                            column_types=p.get("column_types", []),
                            numeric_stats=p.get("numeric_stats", {}),
                            top_n_pairs=p.get("top_n_pairs", []),
                            date_analysis=p.get("date_analysis", []),
                        )
                except Exception as e:
                    profiles[tn] = {"table_name": tn, "error": str(e)}

            profile_msg = {
                "type": "profile_data",
                "profile": profiles,
                "insights": insights,
                "table_count": len(table_names),
            }
            yield f"data: {json.dumps(profile_msg, ensure_ascii=False)}\n\n"

            profile_prompt = f"""你是一个企业数据分析师。根据以下数据画像报告和系统自动发现的数据洞察，用中文给用户一个清晰的数据洞察总结。

用户问题：{request.question}

数据画像内容：
{json.dumps(profiles, ensure_ascii=False, indent=2)}

系统自动发现的数据洞察：
{json.dumps(insights, ensure_ascii=False, indent=2)}

要求：
1. 开头用一两句话总结数据全貌（多少行、多少列、覆盖什么时间范围等）
2. 重点引用系统自动发现的「数据亮点」（相关性分析、帕累托分析、趋势检测的结果），用自然语言解释这些发现意味着什么
3. 按重要性列出3-5个关键发现（数据趋势、分布特征、异常情况）
4. 如果有异常数据（Z-score > 3），明确指出异常值，这些是系统精确计算的结果，直接引用
5. 如果有时间维度数据，说明时间趋势（上升/下降/稳定）
6. 结尾给出1-2条针对性的业务建议
7. 用 Markdown 格式，重要数字用 **加粗**"""

            yield f"data: {json.dumps({'type': 'mode', 'mode': 'profile'}, ensure_ascii=False)}\n\n"

            answer_parts = []
            for chunk in engine.llm.stream(profile_prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"

            full_answer = compress_markdown("".join(answer_parts))
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'sources': [f'[数据画像] {tn}' for tn in table_names], 'mode': 'profile', 'profile': profiles}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
            return

        sql_result = None
        if engine.sql_engine.can_handle(request.question):
            sql_result = engine.sql_engine.query(request.question)

        if sql_result and sql_result.get("success"):
            sql_data = {
                "sql": sql_result.get("sql", ""),
                "columns": sql_result.get("columns", []),
                "rows": sql_result.get("rows", []),
                "row_count": sql_result.get("row_count", 0),
            }
            aggregates = sql_result.get("aggregates", {})
            yield f"data: {json.dumps({'type': 'sql_data', 'mode': 'sql', 'sql_data': sql_data, 'aggregates': aggregates}, ensure_ascii=False)}\n\n"

            answer_parts = []
            for token in engine.sql_engine.stream_generate_answer(request.question, sql_result):
                answer_parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"

            full_answer = compress_markdown("".join(answer_parts))
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'sources': ['[结构化数据查询]'], 'mode': 'sql', 'sql_data': sql_data}, ensure_ascii=False)}\n\n"
        else:
            context_docs = engine.vector_store.search(request.question, top_k=5)
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_classic.chains.combine_documents import create_stuff_documents_chain

            prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "{input}"),
            ])
            chain = create_stuff_documents_chain(engine.llm, prompt)

            yield f"data: {json.dumps({'type': 'mode', 'mode': 'rag'}, ensure_ascii=False)}\n\n"

            answer_parts = []
            async for chunk in chain.astream({"input": request.question, "context": context_docs}):
                if isinstance(chunk, str):
                    token = chunk
                elif isinstance(chunk, dict):
                    token = chunk.get("answer", "") or chunk.get("output", "") or chunk.get("text", "")
                else:
                    token = str(chunk)

                if token:
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"

            full_answer = compress_markdown("".join(answer_parts))

            sources = []
            seen = set()
            for doc in context_docs:
                fn = doc.metadata.get("file_name", "未知")
                if fn not in seen:
                    sources.append(fn)
                    seen.add(fn)

            if sql_result and not sql_result.get("success"):
                full_answer = f"（尝试了数据查询但失败: {sql_result.get('error', '')}）\n\n回退到文档检索结果：\n\n{full_answer}"

            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'sources': sources, 'mode': 'rag'}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents")
async def list_documents():
    from src.core.vectorstore import VectorStore

    store = VectorStore()
    files = store.list_files()
    return {"documents": files}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    from src.core.ingestion import IngestionPipeline

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}，支持: {list(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(content)

    pipeline = IngestionPipeline()
    try:
        result = pipeline.ingest(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档摄入失败: {str(e)}")

    if result.get("table_count", 0) > 0:
        from src.core.table_store import TableStore
        try:
            store = TableStore()
            rels = store.detect_relationships()
        except Exception:
            rels = []

    return {
        "success": True,
        "file_name": result["file_name"],
        "chunk_count": result["chunk_count"],
        "vector_count": result["vector_count"],
        "table_count": result.get("table_count", 0),
    }


@router.post("/table/relationships/detect")
async def detect_table_relationships():
    from src.core.table_store import TableStore

    store = TableStore()
    relationships = store.detect_relationships()
    return {"relationships": relationships, "count": len(relationships)}


@router.get("/table/relationships")
async def get_table_relationships():
    from src.core.table_store import TableStore

    store = TableStore()
    relationships = store.get_relationships()
    return {"relationships": relationships, "count": len(relationships)}


@router.get("/config/tables", response_model=TableInfoResponse)
async def get_table_info():
    from src.core.rag_engine import RAGEngine

    engine = RAGEngine()
    info = engine.get_table_info()
    return TableInfoResponse(tables=info["tables"], files=info["files"])


@router.post("/table/profile")
async def profile_table(request: ProfileRequest):
    from src.core.data_profiler import profile_table
    from src.core.table_store import TableStore

    store = TableStore()
    tables = store.get_table_names()
    if not tables:
        raise HTTPException(status_code=404, detail="没有可用的数据表")

    table_name = request.table_name
    if table_name not in tables:
        raise HTTPException(status_code=404, detail=f"表不存在: {table_name}")

    try:
        result = profile_table(store, table_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据画像生成失败: {str(e)}")


@router.post("/table/profile/all")
async def profile_all_tables():
    from src.core.data_profiler import profile_table
    from src.core.table_store import TableStore

    store = TableStore()
    tables = store.get_table_names()
    if not tables:
        raise HTTPException(status_code=404, detail="没有可用的数据表")

    results = {}
    for table_name in tables:
        try:
            results[table_name] = profile_table(store, table_name)
        except Exception as e:
            results[table_name] = {"error": str(e)}

    return {"tables": results}


@router.delete("/documents/{file_name}")
async def delete_document(file_name: str):
    from src.core.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()
    files = pipeline.list_files()

    if file_name not in files:
        raise HTTPException(status_code=404, detail=f"文档不存在: {file_name}")

    count = pipeline.delete_file(file_name)

    file_path = UPLOAD_DIR / file_name
    if file_path.exists():
        file_path.unlink()

    return {"success": True, "file_name": file_name, "deleted_vectors": count}


@router.delete("/documents")
async def clear_all_documents():
    from src.core.vectorstore import VectorStore
    from src.core.table_store import TableStore

    upload_dir = UPLOAD_DIR
    deleted_files = 0
    deleted_vectors = 0
    deleted_tables = 0

    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                    deleted_files += 1
                except PermissionError:
                    pass

    vs = VectorStore()
    files = vs.list_files()
    for fname in files:
        n = vs.delete_by_file(fname)
        deleted_vectors += n

    ts = TableStore()
    tbl_files = ts.list_files()
    for fname in tbl_files:
        n = ts.delete_by_file(fname)
        deleted_tables += n

    return {
        "success": True,
        "deleted_files": deleted_files,
        "deleted_vectors": deleted_vectors,
        "deleted_tables": deleted_tables,
    }


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return key[:2] + "****" + key[-2:]
    return key[:4] + "****" + key[-4:]


def _read_env() -> dict[str, str]:
    env_path = PROJECT_ROOT / ".env"
    result: dict[str, str] = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    m = re.match(r"^([A-Z_]+)\s*=\s*(.*)$", line)
                    if m:
                        result[m.group(1)] = m.group(2).strip()
    return result


def _write_env(updates: dict[str, str]) -> None:
    env_path = PROJECT_ROOT / ".env"
    existing: dict[str, str] = {}
    lines: list[str] = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                m = re.match(r"^([A-Z_]+)\s*=\s*(.*)$", stripped) if stripped and not stripped.startswith("#") else None
                if m:
                    key = m.group(1)
                    if key in updates:
                        lines.append(f"{key}={updates[key]}\n")
                        existing[key] = updates[key]
                        continue
                    existing[key] = m.group(2).strip()
                lines.append(line)
    for key, value in updates.items():
        if key not in existing:
            lines.append(f"{key}={value}\n")
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    for i in range(len(lines) - 1):
        if not lines[i].endswith("\n"):
            lines[i] += "\n"
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@router.get("/config")
async def get_config():
    _init()
    llm_cfg = get_llm_config()
    emb_cfg = get_embedding_config()
    yaml_cfg = get_yaml_config()
    llm_providers = yaml_cfg.get("llm_providers", {})
    emb_providers = yaml_cfg.get("embedding_providers", {})

    env = _read_env()
    llm_api_key = env.get("LLM_API_KEY", "")
    emb_api_key = env.get("EMBEDDING_API_KEY", "")

    return {
        "llm": {
            "provider": llm_cfg["provider"],
            "base_url": llm_cfg["base_url"],
            "api_key": llm_api_key,
            "api_key_masked": _mask_key(llm_api_key) if llm_api_key else "",
            "model_name": llm_cfg["model_name"],
            "available_providers": llm_providers,
        },
        "embedding": {
            "provider": emb_cfg["provider"],
            "base_url": emb_cfg["base_url"],
            "api_key": emb_api_key,
            "api_key_masked": _mask_key(emb_api_key) if emb_api_key else "",
            "model_name": emb_cfg["model_name"],
            "available_providers": emb_providers,
        },
    }


@router.put("/config")
async def update_config(update: ConfigUpdate):
    env_updates: dict[str, str] = {}

    yaml_cfg = get_yaml_config()
    llm_providers = yaml_cfg.get("llm_providers", {})
    emb_providers = yaml_cfg.get("embedding_providers", {})

    if update.llm_provider is not None:
        env_updates["LLM_PROVIDER"] = update.llm_provider
        provider_cfg = llm_providers.get(update.llm_provider, {})
        if update.llm_base_url is None and provider_cfg.get("base_url"):
            env_updates["LLM_BASE_URL"] = provider_cfg["base_url"]
        if update.llm_model_name is None:
            models = provider_cfg.get("models", [])
            if models:
                env_updates["LLM_MODEL_NAME"] = models[0]

    if update.llm_base_url is not None:
        env_updates["LLM_BASE_URL"] = update.llm_base_url

    if update.llm_api_key is not None and update.llm_api_key:
        env_updates["LLM_API_KEY"] = update.llm_api_key

    if update.llm_model_name is not None:
        env_updates["LLM_MODEL_NAME"] = update.llm_model_name

    if update.embedding_provider is not None:
        env_updates["EMBEDDING_PROVIDER"] = update.embedding_provider
        provider_cfg = emb_providers.get(update.embedding_provider, {})
        if update.embedding_model_name is None:
            models = provider_cfg.get("models", [])
            if models:
                env_updates["EMBEDDING_MODEL_NAME"] = models[0]

    if update.embedding_api_key is not None and update.embedding_api_key:
        env_updates["EMBEDDING_API_KEY"] = update.embedding_api_key

    if update.embedding_model_name is not None:
        env_updates["EMBEDDING_MODEL_NAME"] = update.embedding_model_name

    if env_updates:
        _write_env(env_updates)
        from src.config.settings import reload_config
        reload_config()

    return {"success": True, "updated": list(env_updates.keys())}


class TestLlmRequest(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None


class TestEmbeddingRequest(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None


@router.post("/config/test-llm")
async def test_llm(body: TestLlmRequest = TestLlmRequest()):
    from langchain_openai import ChatOpenAI

    llm_cfg = get_llm_config()

    api_key = body.api_key or llm_cfg.get("api_key", "")
    base_url = body.base_url or llm_cfg.get("base_url", "")
    model_name = body.model_name or llm_cfg.get("model_name", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="未提供 API Key")

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=50,
            timeout=15,
        )
        resp = llm.invoke("回复 OK")
        return {"success": True, "response": resp.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 连接失败: {str(e)}")


@router.post("/config/test-embedding")
async def test_embedding(body: TestEmbeddingRequest = TestEmbeddingRequest()):
    from langchain_openai import OpenAIEmbeddings

    emb_cfg = get_embedding_config()

    api_key = body.api_key or emb_cfg.get("api_key", "")
    base_url = body.base_url or emb_cfg.get("base_url", "")
    model_name = body.model_name or emb_cfg.get("model_name", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="未提供 Embedding API Key")

    try:
        emb = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout=15,
        )
        result = emb.embed_query("test")
        return {"success": True, "dimensions": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding 连接失败: {str(e)}")