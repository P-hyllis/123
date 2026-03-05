# api/routers/evaluate.py
from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/v1/evaluate", tags=["evaluate"])


@dataclass
class _Job:
    id: str
    status: str  # queued|running|succeeded|failed
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    cmd: list[str] = None  # type: ignore


_JOBS: dict[str, _Job] = {}
_LOCK = threading.Lock()


class EvalRunReq(BaseModel):
    # 直接传 evaluate.py 的 CLI 参数列表，比如:
    # ["--dataset", "xxx", "--output", "out.json"]
    args: list[str] = Field(default_factory=list)
    python: str = "python"
    timeout_sec: int = 0  # 0 表示不设超时


class EvalRunResp(BaseModel):
    job_id: str


class EvalJobResp(BaseModel):
    id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    stdout: str
    stderr: str
    cmd: list[str]


def _run_job(job_id: str, python_bin: str, args: list[str], timeout_sec: int):
    with _LOCK:
        job = _JOBS[job_id]
        job.status = "running"
        job.started_at = time.time()

    cmd = [python_bin, "evaluate.py", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
            check=False,
        )
        with _LOCK:
            job.cmd = cmd
            job.stdout = proc.stdout[-200_000:]  # 防止内存爆：最多保留 200k
            job.stderr = proc.stderr[-200_000:]
            job.returncode = proc.returncode
            job.finished_at = time.time()
            job.status = "succeeded" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as e:
        with _LOCK:
            job.cmd = cmd
            job.stdout = (e.stdout or "")[-200_000:]
            job.stderr = (e.stderr or "")[-200_000:] + "\nTIMEOUT"
            job.returncode = None
            job.finished_at = time.time()
            job.status = "failed"
    except Exception as e:
        with _LOCK:
            job.cmd = cmd
            job.stderr = f"{job.stderr}\n{e}"
            job.finished_at = time.time()
            job.status = "failed"


@router.post("/run", response_model=EvalRunResp)
def run_eval(req: EvalRunReq):
    job_id = uuid.uuid4().hex
    job = _Job(
        id=job_id,
        status="queued",
        created_at=time.time(),
        cmd=[req.python, "evaluate.py", *req.args],
    )
    with _LOCK:
        _JOBS[job_id] = job

    t = threading.Thread(
        target=_run_job,
        args=(job_id, req.python, req.args, req.timeout_sec),
        daemon=True,
    )
    t.start()

    return EvalRunResp(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=EvalJobResp)
def get_job(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"id": job_id, "status": "not_found", "created_at": 0, "stdout": "", "stderr": "", "cmd": []}
        return asdict(job)