
# 完整脚本：在最后一轮检索后对最后一次查询扩展做 prompt 检查/纠错，然后再检索、评估、输出文件

# full_modified_search_script.py
import multiprocessing  # 引入多进程支持
import logging  # 引入日志记录工具

import argparse  # 处理命令行参数
import os
import json
from tqdm import tqdm  # 进度条库
from transformers import AutoTokenizer  # 用于加载 tokenizer

# 引入 Pyserini 检索相关模块
from pyserini.analysis import JDefaultEnglishAnalyzer, JWhiteSpaceAnalyzer
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.pyclass import autoclass
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
from pyserini.search.lucene.reranker import ClassifierType, PseudoRelevanceClassifierReranker
from pyserini.search import get_qrels, get_qrels_file

import openai  # 仅为兼容，实际你用的是 VLLMCompletion
from utils.common import save_json, save_jsonl, load_json, file_exists   # 通用工具函数：保存/读取 json/jsonl
from utils.eval_utils import TrecEvaluator   # 评估模块：TREC 格式评估
import spacy, random, re  # 文本处理相关：spacy, 随机数，正则表达式

from prompts import get_prompt   # prompt 模块，用于构造查询扩展 prompt
from collections import Counter
import itertools
import math
import string

# spacy 加载
nlp = spacy.load("en_core_web_sm")
spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
from utils.trec_utils import load_trec  # 引入 TREC 格式数据加载工具

# 截断文本到一定长度（词数限制）
def truncate_text(text, max_len=128):
    if not text:
        return ""
    doc = spacy_nlp(text)
    # 以 token 数为准截断，返回原始片段（保证不切断字符）
    toks = [t.text for t in doc]
    toks = toks[:max_len]
    return " ".join(toks)

# 从模型生成结果中提取双引号包围的句子
def extract_key_sentences(response_text):
    """
    从模型返回中提取用双引号包围的句子；如果没有双引号，则按行解析并取非空行。
    返回一个 list[str]（原始文本去首尾空格）。
    """
    if response_text is None:
        return []
    if isinstance(response_text, dict) and 'content' in response_text:
        text = response_text['content']
    else:
        text = response_text if isinstance(response_text, str) else str(response_text)

    # 优先用双引号提取
    pattern = r'"([^"]{2,})"'
    matches = re.findall(pattern, text)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    # 否则按行拆分，过滤掉空行、提示语、编号等
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    for ln in lines:
        # 忽略明显的 instruction 和元信息
        if ln.lower().startswith(("your task", "original query", "top retrieved")):
            continue
        # 去掉可能的编号前缀 "1. ..." 或 "- "
        ln = re.sub(r'^\s*[\d\-\.\)]+\s*', '', ln)
        if len(ln) >= 2:
            candidates.append(ln)
    return candidates

# 提取最后的 answer（以 </think> 结尾的格式）
def extract_answer(response):
    """从 response（str 或 dict 或 list 中的单项）提取主体文本（支持你原来的 </think> 形式）"""
    if response is None:
        return ""
    if isinstance(response, dict) and 'content' in response:
        text = response['content']
    else:
        text = response if isinstance(response, str) else str(response)
    # 如果包含 </think>，取其后的部分；否则返回原文
    if "</think>" in text:
        return text.split("</think>\n")[-1].strip()
    return text.strip()

# 从一系列 response 中提取 answer
def extract_expansions(response_list):
    """把模型返回的 response_list（可能是 list[str] 或 list[dict]）解析成扩展文本列表（去空、去重）"""
    expansions = []
    for r in response_list:
        txt = extract_answer(r)
        # 先用 extract_key_sentences 提取双引号或行
        items = extract_key_sentences(txt)
        if not items and txt:
            items = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        for it in items:
            it_clean = it.strip()
            if it_clean and it_clean not in expansions:
                expansions.append(it_clean)
    return expansions

# 提取概念
def extract_concepts_from_query(query, top_k=1):
    """优化的概念提取函数"""
    doc = nlp(query)
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
    entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
    concepts = list(dict.fromkeys(noun_chunks + entities))
    return concepts[:top_k]

def filter_overlap_terms(expansions, original_query=None):
    """去掉扩展中完全重复的词或短语，可选地过滤与原查询重叠的内容"""
    if original_query:
        original_terms = set(re.findall(r'\w+', original_query.lower()))
        seen = set()
        filtered = []
        for e in expansions:
            e_clean = e.lower().strip()
            e_terms = set(re.findall(r'\w+', e_clean))
            # 如果扩展的所有 token 都包含在原 query 中，跳过（避免返回仅重复 query）
            if e_terms.issubset(original_terms):
                continue
            if e_clean not in seen:
                seen.add(e_clean)
                filtered.append(e)
        return filtered
    else:
        seen = set()
        filtered = []
        for e in expansions:
            key = e.lower().strip()
            if key not in seen:
                seen.add(key)
                filtered.append(e)
        return filtered

# progressive_query_rewrite 保持不变（稍作格式兼容）
def progressive_query_rewrite(
        openai_api, query, top_passages,
        max_demo_len=None, index=None,
        expansion_method="", 
        reqeat_weight=None,
        accumulated_query_expansions=None,
        accumulate=False,
        topic_id=None,
        *arg, **kwargs):

    # 截断 top_passages
    if max_demo_len:
        top_passages = [truncate_text(psg, max_demo_len) for psg in top_passages]

    top_passages_str = "\n".join([f"{idx+1}. {psg}" for idx, psg in enumerate(top_passages)])
    # 使用你的 get_prompt（若你的 get_prompt 支持不同 method，直接使用）
    user_prompt = get_prompt(expansion_method, query, top_passages_str)

    print(f"using {expansion_method} for query expansion")
    print("Input message:" + user_prompt)

    messages = [{"role": "user", "content": user_prompt}]
    gen_fn = openai_api.completion_chat
    response_list = gen_fn(messages, *arg, **kwargs)

    print("*" * 100 + "\nR1 trace:\n")    
    print(response_list)
    print("\n" + "*" * 100 + "\n")

    # 提取 LLM 扩展
    query_expansions = extract_expansions(response_list)
    print(f"LLM Query expansions: {query_expansions}")

    # 提取原查询的两个概念
    query_concepts = extract_concepts_from_query(query, top_k=2)
    print(f"Extracted concepts: {query_concepts}")

    # 合并并去重（overlap 过滤）
    query_expansions.extend(query_concepts)
    query_expansions = filter_overlap_terms(query_expansions, query)
    print(f"After merging concepts & filtering overlap: {query_expansions}")

    if accumulate:
        accumulated_query_expansions[topic_id].extend(query_expansions)
        query_expansions = accumulated_query_expansions[topic_id]

    # 重复权重
    if reqeat_weight:
        q_repeat = int(len("\n".join(query_expansions).split()) / (len(query.split()) * reqeat_weight))
        q_repeat = max(q_repeat, 1)
    else:
        q_repeat = 1

    # 最终查询拼接
    new_list = [query] * q_repeat + query_expansions
    user_query = "\n".join(new_list).lower()

    return user_query, response_list, accumulated_query_expansions

# =============================
# 改进版：检查并纠正最后一次扩展（替换你原来的实现）
# =============================
def check_and_correct_expansions(openai_api, topic_id, query, response_list, top_passages,
                                 max_demo_len=None, gen_num=5, temperature=0.6, max_tokens=4096,
                                 reqeat_weight=None, accumulated_query_expansions=None, accumulate=False,
                                 fallback_to_original=True): 
    """
    改进版的检查/纠错：
    - 生成多个 candidate responses（最多 gen_num 次）
    - 将每个 response 解析为若干扩展，做清洗 -> 得到多个 candidate expansion-lists
    - 用 spaCy 计算每个 candidate list 的 'coverage score'（与 top_passages 的语义相似度）和 'novelty penalty'（与原 query 重复度），并加上微量多样性奖励
    - 选取得分最高的一个 candidate list 作为最终纠正结果；再保证合并 query-concepts 等逻辑一致性
    返回：user_query（最终拼接好的查询字符串）， corrected_expansions（list），raw_resp_obj（原始模型响应集合）
    """
    # 1) 预处理 top_passages
    if max_demo_len:
        top_passages = [truncate_text(psg, max_demo_len) for psg in top_passages]
    top_passages_str = "\n".join([f"{idx+1}. {psg}" for idx, psg in enumerate(top_passages)])
    # 2) 合并 last generated 原文，作为上下文参考（原来的 response_list 可能已是 list）
    last_generated = "\n".join([extract_answer(r) for r in response_list]) if response_list else ""

    # 3) 构造检查 prompt（更明确的指令），减少 LLM 随机性
    check_prompt = f"""
You are an assistant that improves query expansions for information retrieval.

Original Query:
{query}

Top retrieved passages:
{top_passages_str}

Candidate expansions (from previous step):
{last_generated}

Please produce diverse candidate sets of expansions (one set per response).
For each set:
 - Provide up to {gen_num} concise expansions (phrases or short clauses).
 - Remove exact duplicates and meaningless fragments.
 - Output each expansion quoted (e.g. "expansion phrase") one per line.
Return the candidate set(s) as plain text.
"""
    messages = [{"role": "user", "content": check_prompt}]

    # 4) 调用 LLM 生成多次 responses（每次生成一套 candidate set）
    raw_candidates = []
    for i in range(max(1, gen_num)):
        try:
            resp = openai_api.completion_chat(messages, temperature=temperature, max_tokens=max_tokens, n=1)
        except Exception as e:
            print(f"[check_and_correct] generation error: {e}")
            resp = ""
        # extract text
        if isinstance(resp, list):
            txt = " ".join([extract_answer(r) for r in resp])
        else:
            txt = extract_answer(resp)
        raw_candidates.append(txt)

    # 5) 解析每个 raw candidate 为 expansion-list（清洗）
    candidate_lists = []
    for txt in raw_candidates:
        txt = txt.strip()
        if not txt:
            continue
        # 优先尝试用双引号解析
        items = extract_key_sentences(txt)
        if items:
            candidate_lists.append(items)
            continue
        # 否则按空行（或分段）拆分为一组
        groups = [g for g in re.split(r'\n\s*\n', txt) if g.strip()]
        if groups:
            for g in groups:
                group_items = [re.sub(r'^\s*[\d\-\.\)]+\s*', '', ln).strip()
                              for ln in g.splitlines() if ln.strip()]
                if group_items:
                    candidate_lists.append(group_items)
            continue
        # 最后按行拆
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if lines:
            candidate_lists.append(lines)

    # 如果模型返回为空或解析失败，回退为使用原 response_list 的解析
    if not candidate_lists:
        fallback_lists = [extract_expansions(response_list)] if response_list else []
        candidate_lists = fallback_lists

    # 6) 定义评分函数：coverage + novelty + diversity（权重调整）
    def score_candidate_list(exp_list):
        """
        简化的评分函数，更专注于检索效果
        """
        if not exp_list:
            return -1e6
        
        # 基础清洗
        cleaned = []
        for e in exp_list:
            s = e.strip().strip(string.punctuation).strip()
            # 最小长度与过滤
            if len(s) >= 3 and not (len(s.split()) <= 1 and len(s) <= 3):
                cleaned.append(s)
        
        if not cleaned:
            return -1e6

        # spaCy doc 转换
        try:
            top_docs = [nlp(p) for p in top_passages] if top_passages else []
            q_doc = nlp(query)
            exp_docs = [nlp(e) for e in cleaned]
        except Exception as e:
            print(f"[score_candidate_list] spacy error: {e}")
            return -1e6

        # 覆盖度（与检索文档的相关性）
        coverage_score = 0.0
        if top_docs:
            for ed in exp_docs:
                try:
                    sims = [ed.similarity(td) for td in top_docs]
                except Exception:
                    sims = [0.0 for _ in top_docs]
                coverage_score += max(sims) if sims else 0.0
            coverage_score = coverage_score / len(exp_docs)
        else:
            # 如果没有 top_docs，使用与 query 的相似度作为 proxy
            coverage_score = sum(ed.similarity(q_doc) for ed in exp_docs) / len(exp_docs)

        # 轻微的新颖性检查（避免完全重复原查询）
        novelty_penalty = 0.0
        for ed in exp_docs:
            sim_to_query = ed.similarity(q_doc)
            # 调低惩罚阈值与惩罚量
            if sim_to_query > 0.98:  # 只有几乎完全重复时轻罚
                novelty_penalty += 0.1
        
        # 鼓励适度的扩展数量（但不要过大）
        quantity_bonus = min(0.05 * len(cleaned), 0.3)
        
        # 多样性：计算 pairwise dissimilarity 的平均并缩放，避免完全同质
        pairwise = []
        for i in range(len(exp_docs)):
            for j in range(i + 1, len(exp_docs)):
                try:
                    pairwise.append(1.0 - exp_docs[i].similarity(exp_docs[j]))
                except Exception:
                    pairwise.append(0.0)
        diversity_bonus = 0.0
        if pairwise:
            avg_dissim = sum(pairwise) / len(pairwise)
            diversity_bonus = min(avg_dissim, 1.0) * 0.2  # 缩放到较小贡献

        # 最终得分：主要看覆盖度，轻微惩罚重复，加上多样性/数量奖励
        score = coverage_score + quantity_bonus + diversity_bonus - novelty_penalty
        return score

    # 7) 对候选集合打分并选取 top one
    scored = []
    for clist in candidate_lists:
        sc = score_candidate_list(clist)
        scored.append((sc, clist))
    # 可能 candidate_lists 很多，选 top 1
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_list = scored[0] if scored else (None, [])

    print(f"[Check] topic {topic_id} cand_count={len(candidate_lists)} best_score={best_score}")

    # 8) 清洗最佳集合并做最终 overlap 过滤（与原 query）
    def clean_and_filter_list(clist, original_query):
        res = []
        orig_terms = set(re.findall(r'\w+', original_query.lower()))
        seen = set()
        for e in clist:
            s = re.sub(r'\s+', ' ', e.strip()).strip()
            s = s.strip(string.punctuation)
            if len(s) < 3:
                continue
            # 只跳过与原 query token 集合完全相同的情况（放宽原先的子集过滤）
            e_terms = set(re.findall(r'\w+', s.lower()))
            if e_terms == orig_terms:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            res.append(s)
        return res

    corrected_expansions = clean_and_filter_list(best_list, query)

    # 9) 提取 query 概念并合并（保底）
    query_concepts = extract_concepts_from_query(query, top_k=2)
    # 将 concept 放在 corrected_expansions 前并再次去重
    merged = []
    seen_lower = set()
    for c in (query_concepts + corrected_expansions):
        if c and c.lower() not in seen_lower:
            merged.append(c)
            seen_lower.add(c.lower())
    corrected_expansions = merged

    # 10) 最终再做一次 filter_overlap_terms（但这次我们不把与原 query 部分重合的都删掉）
    # 调用时传 original_query=None -> 只做去重
    corrected_expansions = filter_overlap_terms(corrected_expansions, original_query=None)

    # 如果结果太少且允许回退，则回退到原始 response_list 提取的扩展
    if fallback_to_original and (not corrected_expansions or len(corrected_expansions) < 3):
        fallback = extract_expansions(response_list)
        if fallback:
            print(f"[Check] topic {topic_id} fallback to original expansions (count={len(fallback)})")
            corrected_expansions = fallback

    # 11) apply accumulate if needed
    if accumulate and accumulated_query_expansions is not None:
        accumulated_query_expansions[topic_id].extend(corrected_expansions)
        corrected_expansions = accumulated_query_expansions[topic_id]

    # 12) compute q_repeat same as progressive_query_rewrite
    if reqeat_weight:
        q_repeat = int(len("\n".join(corrected_expansions).split()) / (len(query.split()) * reqeat_weight)) if corrected_expansions else 1
        q_repeat = max(q_repeat, 1)
    else:
        q_repeat = 1

    print(f"[Check] topic {topic_id} corrected_expansions={corrected_expansions} repeat={q_repeat}")

    # 最终拼接（与 progressive_query_rewrite 保持一致）
    new_list = [query] * q_repeat + corrected_expansions
    user_query = "\n".join(new_list).lower()

    # 返回 raw_resp_obj：把 raw_candidates 拼成一个对象供保存/检查
    raw_resp_obj = {"raw_candidates": raw_candidates, "selected_score": best_score, "selected_list": best_list}
    return user_query, corrected_expansions, raw_resp_obj

# 封装 Pyserini 检索器接口（保持与你的实现一致）
class LuceneSearchInterface(object):
    def __init__(self, args):
        if not args.impact:
            if os.path.exists(args.index):
                searcher = LuceneSearcher(args.index)
            else:
                searcher = LuceneSearcher.from_prebuilt_index(args.index)
        elif args.impact:
            if os.path.exists(args.index):
                searcher = LuceneImpactSearcher(args.index, args.encoder, args.min_idf)
            else:
                searcher = LuceneImpactSearcher.from_prebuilt_index(args.index, args.encoder, args.min_idf)
        else:
            raise AttributeError("No searcher specified!")

        if args.language != 'en':
            searcher.set_language(args.language)

        if not searcher:
            exit()

        search_rankers = []
        if args.qld:
            search_rankers.append('qld')
            searcher.set_qld()
        elif args.bm25:
            search_rankers.append('bm25')
            if not args.disable_bm25_param:
                self.set_bm25_parameters(searcher, args.index, args.k1, args.b)

        if args.rm3:
            search_rankers.append('rm3')
            searcher.set_rm3()

        if args.rocchio:
            search_rankers.append('rocchio')
            if args.rocchio_use_negative:
                searcher.set_rocchio(gamma=0.15, use_negative=True)
            else:
                searcher.set_rocchio()
        
        fields = dict()
        if args.fields:
            fields = dict([pair.split('=') for pair in args.fields])
            print(f'Searching over fields: {fields}')

        query_generator = None
        if args.dismax:
            query_generator = autoclass("io.anserini.search.query.DisjunctionMaxQueryGenerator")(args.tiebreaker)
            print(f'Using dismax query generator with tiebreaker={args.tiebreaker}')

        if args.pretokenized:
            analyzer = JWhiteSpaceAnalyzer()
            searcher.set_analyzer(analyzer)
            if args.tokenizer is not None:
                raise ValueError(f"--tokenizer is not supported with when setting --pretokenized.")

        tokenizer = None
        if args.tokenizer != None:
            analyzer = JWhiteSpaceAnalyzer()
            searcher.set_analyzer(analyzer)
            print(f'Using whitespace analyzer because of pretokenized topics')
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f'Using {args.tokenizer} to preprocess topics')

        if args.stopwords:
            analyzer = JDefaultEnglishAnalyzer.fromArguments('porter', False, args.stopwords)
            searcher.set_analyzer(analyzer)
            print(f'Using custom stopwords={args.stopwords}')

        ranker = None
        use_prcl = args.prcl and len(args.prcl) > 0 and args.alpha > 0
        if use_prcl is True:
            ranker = PseudoRelevanceClassifierReranker(
                searcher.index_dir, args.vectorizer, args.prcl, r=args.r, n=args.n, alpha=args.alpha)

        self.args = args
        self.searcher, self.search_rankers = searcher, search_rankers
        self.fields, self.query_generator, self.tokenizer = fields, query_generator, tokenizer
        self.use_prcl, self.ranker = use_prcl, ranker

    @staticmethod
    def set_bm25_parameters(searcher, index, k1=None, b=None):
        if k1 is not None or b is not None:
            if k1 is None or b is None:
                print('Must set *both* k1 and b for BM25!')
                exit()
            print(f'Setting BM25 parameters: k1={k1}, b={b}')
            searcher.set_bm25(k1, b)
        else:
            # 省略你已有的 index-specific defaults（保持原样）
            if index == 'msmarco-passage' or index == 'msmarco-passage-slim' or index == 'msmarco-v1-passage' or \
                    index == 'msmarco-v1-passage-slim' or index == 'msmarco-v1-passage-full':
                print('MS MARCO passage: setting k1=0.82, b=0.68')
                searcher.set_bm25(0.82, 0.68)
            # 其余保持不变（如你原代码）

    def get_setting_name(self):
        tokens = ['run', '+'.join(self.search_rankers)]
        setting_name = '.'.join(tokens)
        if self.use_prcl is True:
            clf_rankers = []
            for t in self.args.prcl:
                if t == ClassifierType.LR:
                    clf_rankers.append('lr')
                elif t == ClassifierType.SVM:
                    clf_rankers.append('svm')
            r_str = f'prcl.r_{self.args.r}'
            n_str = f'prcl.n_{self.args.n}'
            a_str = f'prcl.alpha_{self.args.alpha}'
            clf_str = 'prcl_' + '+'.join(clf_rankers)
            tokens2 = [self.args.vectorizer, clf_str, r_str, n_str, a_str]
            setting_name += ('-' + '-'.join(tokens2))
        return setting_name

    def do_retrieval(self, text, num_hits, num_thread=12):
        if isinstance(text, str):
            if (self.args.tokenizer != None):
                toks = self.tokenizer.tokenize(text)
                text = ' '
                text = text.join(toks)
            if self.args.impact:
                hits = self.searcher.search(text, num_hits, fields=self.fields)
            else:
                hits = self.searcher.search(text, num_hits, query_generator=self.query_generator, fields=self.fields)
            results = hits
        else:
            batch_topics = text
            pseudo_batch_topic_ids = [str(idx) for idx, _ in enumerate(text)]
            if (self.args.tokenizer != None):
                new_batch_topics = []
                for text in batch_topics:
                    toks = self.tokenizer.tokenize(text)
                    text = ' '
                    text = text.join(toks)
                    new_batch_topics.append(text)
                batch_topics = new_batch_topics
            if self.args.impact:
                results = self.searcher.batch_search(
                    batch_topics, pseudo_batch_topic_ids, num_hits, num_thread, fields=self.fields,
                )
            else:
                results = self.searcher.batch_search(
                    batch_topics, pseudo_batch_topic_ids, num_hits, num_thread,
                    query_generator=self.query_generator, fields=self.fields
                )
            results = [results[id_] for id_ in pseudo_batch_topic_ids]
        return results

    def do_rerank(self, hits):
        if self.use_prcl and len(hits) > (self.args.r + self.args.n):
            docids = [hit.docid.strip() for hit in hits]
            scores = [hit.score for hit in hits]
            scores, docids = self.ranker.rerank(docids, scores)
            docid_score_map = dict(zip(docids, scores))
            for hit in hits:
                hit.score = docid_score_map[hit.docid.strip()]
        return hits

    def do_postprocess(self, hits, topic_id, remove_duplicates=False, remove_query=False):
        if remove_duplicates:
            seen_docids = set()
            dedup_hits = []
            for hit in hits:
                if hit.docid.strip() in seen_docids:
                    continue
                seen_docids.add(hit.docid.strip())
                dedup_hits.append(hit)
            hits = dedup_hits
        if remove_query:
            hits = [hit for hit in hits if hit.docid != topic_id]
            print("remove query")
        return hits

def search_iterator(args, search_api, qid_query_list):
    batch_topics = list()
    batch_topic_ids = list()
    for index, (topic_id, text) in enumerate(tqdm(qid_query_list, total=len(qid_query_list))):
        if args.batch_size <= 1 and args.threads <= 1:
            hits = search_api.do_retrieval(text, args.hits)
            results = [(topic_id, hits)]
        else:
            batch_topic_ids.append(topic_id)
            batch_topics.append(text)
            if (index + 1) % args.batch_size == 0 or index == len(qid_query_list) - 1:
                hits_list = search_api.do_retrieval(batch_topics, args.hits, args.threads)
                results = [(id_, hits) for id_, hits in zip(batch_topic_ids, hits_list)]
                batch_topic_ids.clear(), batch_topics.clear()
            else:
                continue
        for topic_id, hits in results:
            hits = search_api.do_rerank(hits)
            hits = search_api.do_postprocess(
                hits, topic_id, args.remove_duplicates, args.remove_query)
            yield topic_id, hits
        results.clear()

def define_search_args(parser):
    parser.add_argument('--index', type=str, metavar='path to index or index name', required=True,
                        help="Path to Lucene index or name of prebuilt index.")
    parser.add_argument('--impact', action='store_true', help="Use Impact.")
    parser.add_argument('--encoder', type=str, default=None, help="encoder name")
    parser.add_argument('--min-idf', type=int, default=0, help="minimum idf")
    parser.add_argument('--bm25', action='store_true', default=True, help="Use BM25 (default).")
    parser.add_argument('--k1', type=float, help='BM25 k1 parameter.')
    parser.add_argument('--b', type=float, help='BM25 b parameter.')
    parser.add_argument('--rm3', action='store_true', help="Use RM3")
    parser.add_argument('--rocchio', action='store_true', help="Use Rocchio")
    parser.add_argument('--rocchio-use-negative', action='store_true', help="Use nonrelevant labels in Rocchio")
    parser.add_argument('--qld', action='store_true', help="Use QLD")
    parser.add_argument('--language', type=str, help='language code for BM25, e.g. zh for Chinese', default='en')
    parser.add_argument('--pretokenized', action='store_true', help="Boolean switch to accept pre-tokenized topics")
    parser.add_argument('--prcl', type=ClassifierType, nargs='+', default=[],
                        help='Specify the classifier PseudoRelevanceClassifierReranker uses.')
    parser.add_argument('--prcl.vectorizer', dest='vectorizer', type=str,
                        help='Type of vectorizer. Available: TfidfVectorizer, BM25Vectorizer.')
    parser.add_argument('--prcl.r', dest='r', type=int, default=10,
                        help='Number of positive labels in pseudo relevance feedback.')
    parser.add_argument('--prcl.n', dest='n', type=int, default=100,
                        help='Number of negative labels in pseudo relevance feedback.')
    parser.add_argument('--prcl.alpha', dest='alpha', type=float, default=0.5,
                        help='Alpha value for interpolation in pseudo relevance feedback.')
    parser.add_argument('--fields', metavar="key=value", nargs='+',
                        help='Fields to search with assigned float weights.')
    parser.add_argument('--dismax', action='store_true', default=False,
                        help='Use disjunction max queries when searching multiple fields.')
    parser.add_argument('--dismax.tiebreaker', dest='tiebreaker', type=float, default=0.0,
                        help='The tiebreaker weight to use in disjunction max queries.')
    parser.add_argument('--stopwords', type=str, help='Path to file with customstopwords.')

def main():
    from vllm_server.vllm_completion import VLLMCompletion
    JLuceneSearcher = autoclass('io.anserini.search.SimpleSearcher')
    parser = argparse.ArgumentParser(description='Search a Lucene index.')
    define_search_args(parser)
    parser.add_argument('--topics', type=str, metavar='topic_name', required=True,
                        help="Name of topics. Available: robust04, robust05, core17, core18.")
    parser.add_argument('--hits', type=int, metavar='num',
                        required=False, default=1000, help="Number of hits.")
    parser.add_argument('--topics-format', type=str, metavar='format', default=TopicsFormat.DEFAULT.value,
                        help=f"Format of topics. Available: {[x.value for x in list(TopicsFormat)]}")
    parser.add_argument('--output-format', type=str, metavar='format', default=OutputFormat.TREC.value,
                        help=f"Format of output. Available: {[x.value for x in list(OutputFormat)]}")
    parser.add_argument('--output', type=str, metavar='path',
                        help="Path to output file.")
    parser.add_argument('--max-passage', action='store_true',
                        default=False, help="Select only max passage from document.")
    parser.add_argument('--max-passage-hits', type=int, metavar='num', required=False, default=100,
                        help="Final number of hits when selecting only max passage.")
    parser.add_argument('--max-passage-delimiter', type=str, metavar='str', required=False, default='#',
                        help="Delimiter between docid and passage id.")
    parser.add_argument('--batch-size', type=int, metavar='num', required=False,
                        default=1, help="Specify batch size to search the collection concurrently.")
    parser.add_argument('--threads', type=int, metavar='num', required=False,
                        default=1, help="Maximum number of threads to use.")
    parser.add_argument('--tokenizer', type=str, help='tokenizer used to preprocess topics')
    parser.add_argument('--remove-duplicates', action='store_true', default=False, help="Remove duplicate docs.")
    parser.add_argument('--remove_query', action='store_true', default=False, help="Remove query from results list.")
    # new
    parser.add_argument('--disable_bm25_param', action='store_true', default=True, help="Use BM25 (default).")
    parser.add_argument('--qrels', type=str, metavar='qrels_name', default=None,
                        help="In case of the difference between topics and qrels")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save outputs")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite existing output dir")
    parser.add_argument("--openai_api_key", type=str, default="none")
    parser.add_argument("--generation_model", type=str, default="/mnt/nvme0n1/zcl/IR/Think_QE/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--answer_key", type=str, default="contents")
    parser.add_argument("--keep_passage_num", type=int, default=10, help="Number of passages kept for CSQE")
    parser.add_argument('--write_top_passages', action='store_true', help="Save the top retrieved passages")
    parser.add_argument("--gen_num", type=int, default=5, help="Number of query expansions to generate")
    parser.add_argument('--max_demo_len', type=int, default=None, help="Truncation length for each retrieved passage")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens to generate each time")
    parser.add_argument('--expansion_method', type=str, default="r1qe")
    parser.add_argument('--trec_python_path', type=str, default="python3")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for query expansion")
    parser.add_argument('--reqeat_weight', type=float, default=3, help="Weight for query repetition of MUGI.")
    parser.add_argument('--accumulate', type=lambda x: x.lower() == 'true', default=False, help="Accumulate query expansions")
    parser.add_argument('--use_passage_filter', type=lambda x: x.lower() == 'true', default=False, help="Use filter for dropping previous seen passages")
    parser.add_argument('--no_thinking', type=lambda x: x.lower() == 'true', default=False, help="No thinking mode for the R1-distill-qwen model")
    parser.add_argument('--num_interaction', type=int, default=3, help="Number of interaction rounds with the corpus")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # Create output directory if needed

    # generation model (your VLLM wrapper)
    openai_api = VLLMCompletion(model_name=args.generation_model, control_thinking=args.no_thinking)

    # query data
    query_iterator = get_query_iterator(args.topics, TopicsFormat(args.topics_format))
    if "bright" in args.topics:
        use_bright = True
    else:
        use_bright = False

    if use_bright:
        qrels_name = args.qrels
        org_qid_query_list = [(k, v) for k, v in query_iterator]
    else:
        qrels_name = args.qrels or args.topics
        qrels = get_qrels(qrels_name)
        org_qid_query_list = [(k, v) for k, v in query_iterator if k in qrels]
    print(f"The number of query is {len(org_qid_query_list)}")

    # my own api
    search_api = LuceneSearchInterface(args)
    trec_eval = TrecEvaluator(args.trec_python_path)

    # round 1 - regular bm25
    last_top_passages = dict()  # qid2passages
    iter_org_qid_query_list = org_qid_query_list

    flag_gen_aug = False
    accumulated_query_expansions = {}
    seen_passages = {}
    last_top_k_passages = {}
    last_last_top_k_passages = {}

    # prepare per-query structures
    for index, (topic_id, query) in enumerate(tqdm(org_qid_query_list, total=len(org_qid_query_list))):
        accumulated_query_expansions[topic_id] = []
        seen_passages[topic_id] = set()
        last_top_k_passages[topic_id] = set()
        last_last_top_k_passages[topic_id] = set()

    # 重要：保存"最后一次"生成的 responses（用于后续检查/纠错）
    last_round_responses = {}  # topic_id -> response_list

    round_num = args.num_interaction + 1
    for ridx in range(round_num):
        output_path_bm25 = os.path.join(args.output_dir, f'bm25-aug{ridx}_result_retrieval.trec')
        if flag_gen_aug:
            aug_qid_query_list = []
            qid2responses = dict()

            for index, (topic_id, query) in enumerate(tqdm(org_qid_query_list, total=len(org_qid_query_list))):
                all_passages = last_top_passages.get(topic_id, [])
                filtered_passages = []
                if args.use_passage_filter:
                    for passage in all_passages:
                        if passage in seen_passages[topic_id]:
                            continue
                        if passage in last_last_top_k_passages[topic_id]:
                            seen_passages[topic_id].add(passage)
                            continue
                        filtered_passages.append(passage)
                        if len(filtered_passages) >= args.keep_passage_num:
                            break
                    top_passages = filtered_passages[:args.keep_passage_num]
                else:
                    top_passages = all_passages[:args.keep_passage_num]

                query_aug, response_list, accumulated_query_expansions = progressive_query_rewrite(
                    openai_api, query, top_passages, n=args.gen_num,
                    max_demo_len=args.max_demo_len, index=args.index,
                    expansion_method=args.expansion_method,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    reqeat_weight=args.reqeat_weight,
                    accumulated_query_expansions=accumulated_query_expansions,
                    accumulate=args.accumulate,
                    topic_id=topic_id
                )
                aug_qid_query_list.append((topic_id, query_aug))
                print(topic_id, "|||", query_aug)

                qid2responses[topic_id] = response_list
                # 同步更新全局 last_round_responses（保持为最新一轮）
                last_round_responses[topic_id] = response_list

            save_json(qid2responses, output_path_bm25 + ".responses.json")
            iter_org_qid_query_list = aug_qid_query_list

        # save query
        save_jsonl(iter_org_qid_query_list, output_path_bm25 + ".topics.jsonl")

        output_writer = get_output_writer(
            output_path_bm25, OutputFormat(args.output_format),
            'w', max_hits=args.hits, tag='Anserini', topics=query_iterator.topics, use_max_passage=args.max_passage,
            max_passage_delimiter=args.max_passage_delimiter, max_passage_hits=args.max_passage_hits)

        with output_writer:
            for topic_id, hits in search_iterator(args, search_api, iter_org_qid_query_list):
                output_writer.write(topic_id, hits)
                # save passage result
                passages = []
                for hit in hits:
                    doc = search_api.searcher.doc(hit.docid)
                    raw_df = json.loads(doc.raw())
                    text_list = [raw_df.get(k) for k in args.answer_key.split("|") if raw_df.get(k)]
                    passages.append("\t".join(text_list))
                last_top_passages[topic_id] = passages

                # Update passage history
                if flag_gen_aug:
                    last_last_top_k_passages[topic_id] = last_top_k_passages[topic_id]
                    last_top_k_passages[topic_id] = set(passages[:args.keep_passage_num])
                else:
                    last_top_k_passages[topic_id] = set(passages[:args.keep_passage_num])

        if args.write_top_passages:
            save_json(last_top_passages, output_path_bm25 + "top-psgs.json")

        if use_bright:
            result_metrics = trec_eval.predefined_bright_trec(qrels_name, output_path_bm25)
        else:
            result_metrics = trec_eval.predefined_msmarco_trec(qrels_name, output_path_bm25)
        save_json(result_metrics, output_path_bm25 + ".metrics.json")
        print(f"At round {ridx}: {json.dumps(result_metrics)}")

        flag_gen_aug = True

    # =======================
    # 所有轮次完成后：对"最后一次"扩展进行检查/纠错 -> 然后再检索 & 输出 & 评估
    # =======================
    print("All rounds finished. Proceeding to check & correct last-round expansions and run final retrieval...")

    # prepare final output filenames
    checked_output_path = os.path.join(args.output_dir, f'bm25-checked_result_retrieval.trec')
    checked_responses = {}

    # Build augmented qid-query list using corrected expansions
    final_aug_qid_query_list = []

    for (topic_id, query) in tqdm(org_qid_query_list, total=len(org_qid_query_list)):
        # get top passages from last_top_passages
        top_passages = last_top_passages.get(topic_id, [])[:args.keep_passage_num]
        # get last-round responses for this topic
        response_list = last_round_responses.get(topic_id, [])
        # call check_and_correct_expansions with all necessary parameters
        user_query, corrected_expansions, checked_resp_obj = check_and_correct_expansions(
            openai_api, topic_id, query, response_list, top_passages,
            max_demo_len=args.max_demo_len, gen_num=args.gen_num,
            temperature=args.temperature, max_tokens=args.max_tokens,
            reqeat_weight=args.reqeat_weight,
            accumulated_query_expansions=accumulated_query_expansions,
            accumulate=args.accumulate
        )
        # store corrected expansions for saving (之前代码有误，这里改为保存 corrected_expansions)
        checked_responses[topic_id] = corrected_expansions if corrected_expansions is not None else []

        # 使用 check_and_correct_expansions 返回的完整查询字符串
        final_aug_qid_query_list.append((topic_id, user_query))

    # 保存 checked responses
    save_json(checked_responses, checked_output_path + ".responses.json")
    # 保存 topics.jsonl
    save_jsonl(final_aug_qid_query_list, checked_output_path + ".topics.jsonl")

    # 写 trec 并评估
    output_writer = get_output_writer(
        checked_output_path, OutputFormat(args.output_format),
        'w', max_hits=args.hits, tag='Anserini-checked', topics=query_iterator.topics, use_max_passage=args.max_passage,
        max_passage_delimiter=args.max_passage_delimiter, max_passage_hits=args.max_passage_hits)

    with output_writer:
        for topic_id, hits in search_iterator(args, search_api, final_aug_qid_query_list):
            output_writer.write(topic_id, hits)
            passages = []
            for hit in hits:
                doc = search_api.searcher.doc(hit.docid)
                raw_df = json.loads(doc.raw())
                text_list = [raw_df.get(k) for k in args.answer_key.split("|") if raw_df.get(k)]
                passages.append("\t".join(text_list))
            # save top passages for checked run as well
            last_top_passages[topic_id] = passages
    # 保存 top passages 文件（若需要）
    if args.write_top_passages:
        save_json(last_top_passages, checked_output_path + "top-psgs.json")

    # eval
    if use_bright:
        result_metrics_checked = trec_eval.predefined_bright_trec(qrels_name, checked_output_path)
    else:
        result_metrics_checked = trec_eval.predefined_msmarco_trec(qrels_name, checked_output_path)
    save_json(result_metrics_checked, checked_output_path + ".metrics.json")
    print(f"Checked final retrieval metrics: {json.dumps(result_metrics_checked)}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
