from fastapi import FastAPI, HTTPException
from api.schemas import SingleRunRequest, SingleRunResponse, BatchRunRequest
from api.service import run_single, run_batch

app = FastAPI(title="REA-BM25 (QE + Check) API", version="0.1")

@app.post("/v1/run/single", response_model=SingleRunResponse)
def api_single(req: SingleRunRequest):
    try:
        return run_single(
            cfg=req.config.model_dump(by_alias=True),
            qid=req.qid,
            query=req.query,
            do_check=req.do_check,
            topk_for_ui=req.topk_for_ui
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/run/batch")
def api_batch(req: BatchRunRequest):
    try:
        return run_batch(
            cfg=req.config.model_dump(by_alias=True),
            queries=req.queries,
            run_eval=req.run_eval
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))