# api/routers/rag.py
from __future__ import annotations

import json
from typing import Any, Generator

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.deps import get_rag_service
from services.rag_service_stream import RAGService


router = APIRouter(prefix="/v1/rag", tags=["rag"])


class ChatReq(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResp(BaseModel):
    answer: str


class UploadRespItem(BaseModel):
    filename: str
    success: bool
    message: str


class UploadResp(BaseModel):
    results: list[UploadRespItem]


class _UploadAdapter:
    """
    适配 RAGService.process_document 需要的接口：
      - .name
      - .getvalue() -> bytes
    """
    def __init__(self, filename: str, content: bytes):
        self.name = filename
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


@router.post("/documents", response_model=UploadResp)
async def upload_documents(
    files: list[UploadFile] = File(...),
    rag: RAGService = Depends(get_rag_service),
):
    results: list[UploadRespItem] = []

    for f in files:
        content = await f.read()  # 一次性读入内存；大文件可改成落盘再读
        adapter = _UploadAdapter(f.filename, content)
        out: dict[str, Any] = rag.process_document(adapter)  # process_document 依赖 name/getvalue() :contentReference[oaicite:4]{index=4}
        results.append(
            UploadRespItem(
                filename=f.filename,
                success=bool(out.get("success")),
                message=str(out.get("message", "")),
            )
        )

    return UploadResp(results=results)


@router.delete("/database")
def clear_database(rag: RAGService = Depends(get_rag_service)):
    ok = rag.clear_database()
    return {"ok": ok}


@router.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, rag: RAGService = Depends(get_rag_service)):
    ans = rag.get_answer(req.question)  # get_answer 会汇总 get_answer_stream :contentReference[oaicite:5]{index=5}
    return ChatResp(answer=ans or "")


@router.post("/chat/stream")
def chat_stream(req: ChatReq, rag: RAGService = Depends(get_rag_service)):
    """
    SSE 流式返回：event: token
    """
    def gen() -> Generator[bytes, None, None]:
        try:
            for chunk in rag.get_answer_stream(req.question):  # :contentReference[oaicite:6]{index=6}
                payload = {"delta": chunk}
                # SSE 格式：data: <json>\n\n
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: {\"done\": true}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")