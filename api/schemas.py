from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class RunConfig(BaseModel):
    # ====== Lucene / Pyserini 参数（对齐 rea_bm25.py 的 argparse）======
    index: str
    impact: bool = False
    encoder: Optional[str] = None
    min_idf: int = Field(0, alias="min-idf")
    bm25: bool = True
    k1: Optional[float] = None
    b: Optional[float] = None
    rm3: bool = False
    rocchio: bool = False
    rocchio_use_negative: bool = Field(False, alias="rocchio-use-negative")
    qld: bool = False
    language: str = "en"
    pretokenized: bool = False
    fields: Optional[List[str]] = None
    dismax: bool = False
    tiebreaker: float = 0.0
    stopwords: Optional[str] = None
    tokenizer: Optional[str] = None
    remove_duplicates: bool = False
    remove_query: bool = False
    disable_bm25_param: bool = True

    # ====== 检索与批处理 ======
    hits: int = 1000
    batch_size: int = 1
    threads: int = 1

    # ====== LLM QE / check 参数（对齐 rea_bm25.py）======
    output_dir: Optional[str] = None
    overwrite_output_dir: bool = False
    generation_model: str
    answer_key: str = "contents"
    keep_passage_num: int = 10
    write_top_passages: bool = False
    gen_num: int = 5
    max_demo_len: Optional[int] = None
    max_tokens: int = 32768
    expansion_method: str = "r1qe"
    trec_python_path: str = "python3"
    temperature: float = 0.6
    reqeat_weight: float = 3.0
    accumulate: bool = False
    use_passage_filter: bool = False
    no_thinking: bool = False
    num_interaction: int = 3

    # ====== batch 输入（topics/qrels）======
    topics: Optional[str] = None
    topics_format: str = "default"
    output_format: str = "trec"
    qrels: Optional[str] = None

class SingleRunRequest(BaseModel):
    config: RunConfig
    qid: str = "q1"
    query: str
    do_check: bool = True
    topk_for_ui: int = 10

class BatchRunRequest(BaseModel):
    config: RunConfig
    # 两种方式：1) 直接传 queries；2) 走你原来的 topics 文件
    queries: Optional[List[Dict[str, str]]] = Field(
        None, description='[{"qid":"1","query":"..."}]'
    )
    run_eval: bool = True

class SingleRunResponse(BaseModel):
    qid: str
    original_query: str
    top_passages: List[Dict[str, Any]]
    parsed_expansions: List[str]
    corrected_expansions: List[str]
    final_query: str
    final_hits: List[Dict[str, Any]]
    debug: Dict[str, Any] = {}