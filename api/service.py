from types import SimpleNamespace
from typing import Dict, Any, List, Tuple
import json
import os

# 直接复用仓库里的实现
import rea_bm25  # 注意：这是你仓库根目录的 rea_bm25.py

def to_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    # 兼容字段别名（min-idf / rocchio-use-negative）
    if "min-idf" in cfg and "min_idf" not in cfg:
        cfg["min_idf"] = cfg["min-idf"]
    if "rocchio-use-negative" in cfg and "rocchio_use_negative" not in cfg:
        cfg["rocchio_use_negative"] = cfg["rocchio-use-negative"]

    # 让 args.xxx 可用
    return SimpleNamespace(**cfg)

def build_llm(args: SimpleNamespace):
    from vllm_server.vllm_completion import VLLMCompletion
    return VLLMCompletion(model_name=args.generation_model, control_thinking=args.no_thinking)

def _hits_to_passages(search_api, hits, answer_key: str, limit: int):
    out = []
    for hit in hits[:limit]:
        doc = search_api.searcher.doc(hit.docid)
        raw_df = json.loads(doc.raw())
        text_list = [raw_df.get(k) for k in answer_key.split("|") if raw_df.get(k)]
        out.append({
            "docid": hit.docid,
            "score": float(hit.score),
            "text": "\t".join(text_list) if text_list else ""
        })
    return out

def run_single(cfg: Dict[str, Any], qid: str, query: str, do_check: bool, topk_for_ui: int):
    args = to_args(cfg)
    llm = build_llm(args)
    search_api = rea_bm25.LuceneSearchInterface(args)

    # 1) 初检一次，拿 top passages 给 QE
    iterator = rea_bm25.search_iterator(args, search_api, [(qid, query)])
    _, hits = next(iterator)
    top_passages = _hits_to_passages(search_api, hits, args.answer_key, args.keep_passage_num)

    top_texts = [p["text"] for p in top_passages]

    # 2) progressive QE
    accumulated = {qid: []}
    user_query, response_list, accumulated = rea_bm25.progressive_query_rewrite(
        llm, query, top_texts,
        n=args.gen_num,
        max_demo_len=args.max_demo_len,
        index=args.index,
        expansion_method=args.expansion_method,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reqeat_weight=args.reqeat_weight,
        accumulated_query_expansions=accumulated,
        accumulate=args.accumulate,
        topic_id=qid
    )

    parsed_expansions = rea_bm25.extract_expansions(response_list)

    corrected_expansions = []
    if do_check:
        user_query, corrected_expansions, checked_obj = rea_bm25.check_and_correct_expansions(
            llm, qid, query, response_list, top_texts,
            max_demo_len=args.max_demo_len,
            gen_num=args.gen_num,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reqeat_weight=args.reqeat_weight,
            accumulated_query_expansions=accumulated,
            accumulate=args.accumulate
        )

    # 3) 最终检索
    iterator2 = rea_bm25.search_iterator(args, search_api, [(qid, user_query)])
    _, final_hits = next(iterator2)
    final_hits_payload = _hits_to_passages(search_api, final_hits, args.answer_key, topk_for_ui)

    return {
        "qid": qid,
        "original_query": query,
        "top_passages": top_passages,
        "parsed_expansions": parsed_expansions,
        "corrected_expansions": corrected_expansions,
        "final_query": user_query,
        "final_hits": final_hits_payload,
        "debug": {
            "keep_passage_num": args.keep_passage_num,
            "gen_num": args.gen_num,
            "num_interaction": getattr(args, "num_interaction", None)
        }
    }

def run_batch(cfg: Dict[str, Any], queries: List[Dict[str, str]] | None, run_eval: bool):
    """
    最稳的封装方式：直接用 subprocess 调你原来的 rea_bm25.py CLI，不动原脚本。
    优点：完全不改你实验代码；缺点：批量任务是“外部进程”。
    """
    import subprocess
    args = to_args(cfg)

    if not args.output_dir:
        raise ValueError("batch 模式必须提供 output_dir（对齐你原脚本）")

    # 若没传 queries，就按你原来的 topics 文件跑
    cmd = ["python", "rea_bm25.py"]

    # 下面这些参数来自你 main() 里的 argparse（都在 rea_bm25.py 里）:contentReference[oaicite:3]{index=3}
    def add(flag, val):
        if val is None:
            return
        cmd.extend([flag, str(val)])

    add("--index", args.index)
    if args.impact: cmd.append("--impact")
    add("--encoder", args.encoder)
    add("--min-idf", getattr(args, "min_idf", 0))
    if args.bm25: cmd.append("--bm25")
    add("--k1", args.k1)
    add("--b", args.b)
    if args.rm3: cmd.append("--rm3")
    if args.rocchio: cmd.append("--rocchio")
    if getattr(args, "rocchio_use_negative", False): cmd.append("--rocchio-use-negative")
    if args.qld: cmd.append("--qld")
    add("--language", args.language)

    add("--topics", args.topics)
    add("--topics-format", args.topics_format)
    add("--output-format", args.output_format)
    add("--hits", args.hits)
    add("--batch-size", args.batch_size)
    add("--threads", args.threads)
    if args.remove_duplicates: cmd.append("--remove-duplicates")
    if args.remove_query: cmd.append("--remove_query")
    if args.disable_bm25_param: cmd.append("--disable_bm25_param")

    add("--output_dir", args.output_dir)
    if args.overwrite_output_dir: cmd.append("--overwrite_output_dir")
    add("--generation_model", args.generation_model)
    add("--answer_key", args.answer_key)
    add("--keep_passage_num", args.keep_passage_num)
    if args.write_top_passages: cmd.append("--write_top_passages")
    add("--gen_num", args.gen_num)
    add("--max_demo_len", args.max_demo_len)
    add("--max_tokens", args.max_tokens)
    add("--expansion_method", args.expansion_method)
    add("--trec_python_path", args.trec_python_path)
    add("--temperature", args.temperature)
    add("--reqeat_weight", args.reqeat_weight)
    add("--accumulate", str(args.accumulate).lower())
    add("--use_passage_filter", str(args.use_passage_filter).lower())
    add("--no_thinking", str(args.no_thinking).lower())
    add("--num_interaction", args.num_interaction)
    add("--qrels", args.qrels)

    # 如果你强烈想“传 queries 列表”：
    # 这里可以临时写一个 topics.jsonl 到 output_dir，再把 --topics 指向它（需要你仓库里支持读 jsonl；当前是 get_query_iterator）
    if queries:
        # 先简单落一个 jsonl（qid/query）
        os.makedirs(args.output_dir, exist_ok=True)
        tmp_topics = os.path.join(args.output_dir, "api_topics.jsonl")
        with open(tmp_topics, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(json.dumps({"qid": q["qid"], "query": q["query"]}, ensure_ascii=False) + "\n")
        # ⚠️ 你当前 rea_bm25.py 用 get_query_iterator 读 topics，
        # 如果它不支持 jsonl，你需要在脚本里加一个“jsonl 读取分支”（我下面给你补丁思路）
        # 这里先把路径传过去
        add("--topics", tmp_topics)

    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout": p.stdout[-8000:],  # 防止太长
        "stderr": p.stderr[-8000:],
        "output_dir": args.output_dir
    }