# api/deps.py
from __future__ import annotations

import os
from functools import lru_cache

from services.rag_service_stream import RAGService
from vllm_server.vllm_completion import VLLMCompletion


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    # 按需调整参数
    return RAGService(
        retrieve_k=int(os.getenv("RAG_RETRIEVE_K", "8")),
        enable_reranker=os.getenv("RAG_ENABLE_RERANKER", "true").lower() == "true",
        enable_concept_expansion=os.getenv("RAG_ENABLE_CONCEPT", "false").lower() == "true",
        concept_count=int(os.getenv("RAG_CONCEPT_COUNT", "3")),
        compare_with_raw_query=os.getenv("RAG_COMPARE_RAW", "false").lower() == "true",
    )


@lru_cache(maxsize=1)
def get_vllm() -> VLLMCompletion:
    model_name = os.getenv("VLLM_MODEL", "").strip()
    if not model_name:
        # 兜底：沿用你仓库里 VLLMCompletion 的默认 model_name（代码中写死的本地路径）:contentReference[oaicite:2]{index=2}
        return VLLMCompletion()

    return VLLMCompletion(
        model_name=model_name,
        control_thinking=os.getenv("VLLM_CONTROL_THINKING", "false").lower() == "true",
    )