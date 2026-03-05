# api/routers/llm.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_vllm
from vllm_server.vllm_completion import VLLMCompletion


router = APIRouter(prefix="/v1/llm", tags=["llm"])


class Msg(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class LLMChatReq(BaseModel):
    messages: list[Msg]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1


@router.post("/chat")
def llm_chat(req: LLMChatReq, llm: VLLMCompletion = Depends(get_vllm)):
    out = llm.completion_chat(
        messages=[m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        n=req.n,
    )
    return {"choices": out}