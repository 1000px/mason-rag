import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_env(env_path: Path) -> None:
    if env_path.exists():
        load_dotenv(env_path)


_init_done = False
_yaml_config: dict = {}
_env_overrides: dict = {}


def _init() -> None:
    global _init_done, _yaml_config, _env_overrides
    if _init_done:
        return

    env_path = PROJECT_ROOT / ".env"
    _load_env(env_path)

    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    _yaml_config = _load_yaml(yaml_path)

    _env_overrides = {
        "llm_provider": os.getenv("LLM_PROVIDER"),
        "llm_base_url": os.getenv("LLM_BASE_URL"),
        "llm_api_key": os.getenv("LLM_API_KEY"),
        "llm_model_name": os.getenv("LLM_MODEL_NAME"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER"),
        "embedding_base_url": os.getenv("EMBEDDING_BASE_URL"),
        "embedding_api_key": os.getenv("EMBEDDING_API_KEY"),
        "embedding_model_name": os.getenv("EMBEDDING_MODEL_NAME"),
        "app_host": os.getenv("APP_HOST"),
        "app_port": int(os.getenv("APP_PORT", "8000")),
        "app_debug": os.getenv("APP_DEBUG", "false").lower() == "true",
        "data_dir": os.getenv("DATA_DIR"),
        "upload_dir": os.getenv("UPLOAD_DIR"),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    }

    _init_done = True


def get_yaml_config() -> dict:
    _init()
    return _yaml_config


def get(key: str, default: Any = None) -> Any:
    _init()
    env_val = _env_overrides.get(key)
    if env_val:
        return env_val
    parts = key.split(".")
    val: Any = _yaml_config
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
    return val if val is not None else default


def get_llm_config() -> dict:
    _init()
    provider = _env_overrides.get("llm_provider") or "deepseek"
    providers_cfg = _yaml_config.get("llm_providers", {})
    provider_cfg = providers_cfg.get(provider, {})

    base_url = _env_overrides.get("llm_base_url") or provider_cfg.get("base_url", "")
    api_key = _env_overrides.get("llm_api_key") or ""
    model_name = _env_overrides.get("llm_model_name") or ""

    if not model_name:
        models = provider_cfg.get("models", [])
        if models:
            model_name = models[0]

    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }


def get_embedding_config() -> dict:
    _init()
    llm_cfg = get_llm_config()
    provider = _env_overrides.get("embedding_provider") or llm_cfg["provider"]
    providers_cfg = _yaml_config.get("embedding_providers", {})
    provider_cfg = providers_cfg.get(provider, {})

    base_url = (
        _env_overrides.get("embedding_base_url")
        or provider_cfg.get("base_url", "")
        or llm_cfg["base_url"]
    )
    api_key = _env_overrides.get("embedding_api_key") or llm_cfg["api_key"]
    model_name = _env_overrides.get("embedding_model_name") or ""

    if not model_name:
        models = provider_cfg.get("models", [])
        model_name = models[0] if models else "text-embedding-3-small"

    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }


def get_app_config() -> dict:
    _init()
    return {
        "host": _env_overrides.get("app_host") or "0.0.0.0",
        "port": _env_overrides.get("app_port") or 8000,
        "debug": _env_overrides.get("app_debug") or False,
    }


def get_llm_providers() -> dict:
    _init()
    return _yaml_config.get("llm_providers", {})


def get_embedding_providers() -> dict:
    _init()
    return _yaml_config.get("embedding_providers", {})


def get_document_config() -> dict:
    _init()
    yaml_doc = _yaml_config.get("document", {})
    return {
        "chunk_size": _env_overrides.get("chunk_size") or yaml_doc.get("chunk_size", 1000),
        "chunk_overlap": _env_overrides.get("chunk_overlap") or yaml_doc.get("chunk_overlap", 200),
        "supported_formats": yaml_doc.get("supported_formats", ["pdf", "docx", "xlsx", "txt", "md", "markdown"]),
        "max_upload_size_mb": yaml_doc.get("max_upload_size_mb", 50),
    }


def get_retrieval_config() -> dict:
    _init()
    return _yaml_config.get("retrieval", {
        "top_k": 5,
        "similarity_threshold": 0.7,
        "use_mmr": True,
        "mmr_fetch_k": 10,
        "mmr_lambda_mult": 0.7,
    })


def reload_config() -> None:
    global _init_done, _yaml_config, _env_overrides
    _init_done = False
    _init()


def resolve_path(relative_path: str) -> Path:
    return (PROJECT_ROOT / relative_path).resolve()