

import logging
import os
import re
import tempfile
from typing import Dict, Optional, Any, Generator

from chromadb import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage

from models.langchain_embedding import initialize_embedding_model
from models.langchain_llm import langchain_qwen_llm
from models.reranker_model import RerankerCrossModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 禁用 ChromaDB  遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class RAGService:
    """
    RAG（检索增强生成）服务类，实现文档解析、向量化存储及基于检索的知识进行问答，辅助 LLM 生成更准确、有依据的回答。
    核心流程：文档上传→解析分块→向量化存储→检索相关片段→LLM生成答案。
    支持流式输出。
    """

    def __init__(self,
                 persist_directory: str = "chroma_db",
                 retrieve_k: int = 8,  # 检索 top-k 个相关文本块
                 enable_reranker: bool = True,  # 是否开启重排
                 enable_concept_expansion: bool = False,  # 是否开启概念抽取增强检索
                 concept_count: int = 3,  # 概念抽取个数
                 compare_with_raw_query: bool = False,  # 是否执行“原查询 vs 概念增强查询”双路对比
                 model_name_or_path: str = "BAAI/bge-reranker-large",  # 重排模型名称（HuggingFace Hub规范名）或本地存储路径
                 rerank_top_n: int = 4,  # 重排后保留数量,必须小于retrieve_k
                 rerank_score_threshold: float = 0.1  # 重排分数阈值，大于该阈值才被选取
                 ):
        """
        初始化RAG服务，加载嵌入模型、LLM模型及已存在的向量数据库。

        Args:
            persist_directory: 向量数据库持久化存储路径，默认值为"chroma_db"
            model_name_or_path: 可选，重排模型名称（HuggingFace Hub规范名）或本地存储路径，默认使用BAAI的bge-reranker-v2-m3（中文性价比较高）
                - 1.模型名称：本地缓存（默认~/.cache/huggingface/）无该模型时，自动从Hub下载权重/配置/分词器；缓存已存在则直接加载，无需重复下载。
                - 2.本地路径：需手动下载完整模型文件（包含config.json、model.safetensors/pytorch_model.bin等）到本地路径地址。
            retrieve_k: 可选，向量检索阶段从数据库中召回的候选文本块数量，默认值10。
            rerank_top_n: 可选，从召回的候选文本块中重排筛选后，最终保留的高相关文本块数量，默认值3。约束：必须满足 rerank_top_n < retrieve_k。
            rerank_score_threshold:可选，重排结果的分数筛选阈值，默认值0.1。仅得分超过该阈值的文本块会被保留。

        """
        # 向量数据库持久化目录
        self.persist_directory = persist_directory
        # 初始化嵌入模型（用于将文本转换为向量）
        self.embeddings = initialize_embedding_model("qwen")
        # 检索 top-k 个相关文本块
        self.retrieve_k = retrieve_k
        # 初始化向量数据库（若存在）
        self.vectordb = self._load_vector_db()
        # 初始化大语言模型（用于生成答案）
        self.llm = langchain_qwen_llm()
        # 是否开启重排
        self.enable_reranker = enable_reranker
        # 是否开启概念抽取增强检索
        self.enable_concept_expansion = enable_concept_expansion
        # 概念抽取数量
        self.concept_count = concept_count
        # 是否双路对比
        self.compare_with_raw_query = compare_with_raw_query
        # 初始化重排模型
        self.reranker_model = self._init_rerank_model(model_name_or_path)
        # 重排后保留数量,必须小于k
        self.rerank_top_n = rerank_top_n
        # 重排分数阈值，大于该阈值才被选取
        self.rerank_score_threshold = rerank_score_threshold
        # 保存当前流式回答，用于完整存储
        self.current_stream_answer = ""
        # 初始化内存，设置窗口大小 k=50（只保留最近100轮对话）
        # ConversationBufferWindowMemory 是 ConversationBufferMemory 的扩展版本，专门用于解决长对话场景下的
        # 上下文管理问题。它通过只保留最近的 N 轮对话（滑动窗口机制），在维持对话连贯性的同时，避免历史记录过长导致的 Token 超限问题。
        self.memory = ConversationBufferWindowMemory(
            k=50,  # 窗口大小：仅保留最近50轮对话（1轮=1次用户+1次助手交互）
            return_messages=True,  # 返回LangChain标准Message对象（而非纯字符串，便于格式统一）
            memory_key="chat_history",  # 记忆数据的存储键（后续提取历史时使用）
            output_key="answer",  # 与LLM输出结果的键对齐（适配链式调用规范）
            input_key="input"  # 与LLM输出结果的键对齐（适配链式调用规范）
        )

    def _load_vector_db(self) -> Optional[Chroma]:
        """
        私有方法：加载已持久化的向量数据库（若目录存在且非空）。
        向量数据库用于存储文档片段的向量表示，支持高效的相似性检索。

        Returns:
            加载成功的Chroma向量数据库实例；若不存在或加载失败，返回None

        Raises:
            RuntimeError: 数据库加载过程中发生错误时抛出异常
        """
        # 路径不存在时自动创建（支持多级目录）
        if not os.path.exists(self.persist_directory):
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
            except Exception as e:
                error_msg = f"创建Chroma数据库路径失败：{self.persist_directory}，错误：{str(e)}"
                raise RuntimeError(error_msg) from e

        # 检查持久化目录是否存在且非空
        try:
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        except Exception as e:
            raise RuntimeError(f"向量数据库加载失败（路径：{self.persist_directory}）：{str(e)}")

    # ===================== 重排模型初始化 =====================
    @staticmethod
    def _init_rerank_model(model_name_or_path: str = "BAAI/bge-reranker-large") -> RerankerCrossModel | None:
        """
        初始化重排模型，用于对检索结果进行语义重排序.默认使用BAAI的bge-reranker-large。

        Args:
            model_name_or_path: 重排模型名称（HuggingFace Hub规范名）或本地存储路径，
        Returns:
            CrossEncoder: 初始化后的重排模型实例
        """
        try:
            # 加载重排模型
            rerank_model = RerankerCrossModel(model_name_or_path)
            logger.info(f"成功加载重排模型: {model_name_or_path}")
            return rerank_model
        except Exception as e:
            logger.info(f"加载重排模型失败: {str(e)}")
            return None

    def process_document(self, file: Any) -> Dict[str, bool | str]:
        """
        处理用户上传的文档（解析、分块、向量化、存储到向量数据库）。
        支持的格式：PDF、DOCX、TXT、MD（可通过扩展loader支持更多格式）。

        Args:
            file: 上传的文件对象，需支持以下方法：
                - name: 属性，返回文件名（用于判断格式）
                - getvalue(): 方法，返回文件二进制内容（用于写入临时文件）

        Returns:
            处理结果字典，包含：
                - success: bool，处理是否成功
                - message: str，处理结果描述（成功时含片段数量，失败时含错误信息）
        """
        #  验证文件对象有效性
        if not file or not hasattr(file, 'name') or not hasattr(file, 'getvalue'):
            return {"success": False, "message": "无效的文件对象"}

        # 提取并标准化文件后缀（转为小写，便于格式判断）
        file_name = file.name
        file_suffix = file_name.split('.')[-1].lower() if '.' in file_name else ''
        tmp_file_path = None  # 临时文件路径（用于后续清理）

        try:
            # 创建临时文件存储上传的文件内容（避免直接操作内存中的二进制数据）
            with tempfile.NamedTemporaryFile(
                    delete=False,  # 关闭自动删除，确保加载器能读取
                    suffix=f".{file_suffix}",  # 保留文件后缀，避免加载器解析错误
                    mode='wb'  # 二进制写入模式
            ) as tmp_file:
                tmp_file.write(file.getvalue())  # 写入文件内容
                tmp_file_path = tmp_file.name  # 记录临时文件路径

            # 根据文件后缀选择对应的文档加载器
            if file_suffix == 'pdf':
                loader = PyPDFLoader(tmp_file_path)  # PDF加载器
            elif file_suffix == 'docx':
                loader = Docx2txtLoader(tmp_file_path)  # DOCX加载器
            elif file_suffix in ['txt', 'md']:
                loader = TextLoader(tmp_file_path, encoding='utf-8')  # 文本文件加载器（支持UTF8编码）
            else:
                return {
                    "success": False,
                    "message": f"不支持的文件类型：{file_suffix}，当前支持：pdf/docx/txt/md"
                }

            # 加载文档内容（返回Document对象列表，每个对象含page_content和metadata）
            documents = loader.load()
            if not documents:  # 处理空文档情况
                return {"success": False, "message": "文档加载失败：内容为空或无法解析"}

            # 初始化文本分块器（解决长文本超出模型上下文窗口的问题）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # 每个片段的字符数（根据模型上下文调整）
                chunk_overlap=200,  # 片段间重叠字符数（保持上下文连贯性）
                separators=["\n\n", "\n", "。", " ", ""]  # 优先按中文标点分割，提升分块合理性
            )
            # 将文档分割为片段（每个片段作为独立单元存入向量库）
            splits = text_splitter.split_documents(documents)

            # 将片段添加到向量数据库
            if self.vectordb:
                # 若数据库已存在，直接添加新片段
                self.vectordb.add_documents(splits)
            else:
                # 若数据库不存在，创建新库并添加片段
                self.vectordb = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,  # 使用初始化的嵌入模型
                    persist_directory=self.persist_directory  # 指定存储路径
                )

            return {
                "success": True,
                "message": f"文档处理成功！共添加 {len(splits)} 个文本片段（文件：{file_name}）"
            }

        except Exception as e:  # 捕获所有异常，返回具体错误信息
            return {"success": False, "message": f"文档处理失败（{file_name}）：{str(e)}"}
        finally:
            # 确保临时文件被清理（无论处理成功/失败）
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    logger.error(f"警告：临时文件清理失败（路径：{tmp_file_path}）：{str(e)}")

    def _extract_concepts(self, question: str) -> list[str]:
        """通过LLM从问题中抽取关键概念（数量可配置）。"""
        prompt = f"""
你是一个概念抽取器。
请从用户问题中抽取最关键的 {self.concept_count} 个检索概念，要求：
1. 仅返回概念，不要解释
2. 优先返回名词或短语
3. 每行一个概念
4. 概念数量不超过 {self.concept_count}

用户问题：{question}
"""
        try:
            resp = self.llm.invoke(prompt)
            raw_text = (resp.content or "").strip()
            if not raw_text:
                return []

            concepts = []
            for line in raw_text.splitlines():
                item = line.strip().lstrip("-•0123456789. ").strip()
                if item and item not in concepts:
                    concepts.append(item)
                if len(concepts) >= self.concept_count:
                    break
            return concepts
        except Exception as e:
            logger.warning(f"概念抽取失败，将退化为原始检索：{str(e)}")
            return []

    def _retrieve_docs(self, query: str) -> list:
        """统一检索入口，支持配置检索数量。"""
        retriever = self.vectordb.as_retriever(search_kwargs={"k": self.retrieve_k})
        docs = retriever.invoke(query)
        return docs

    def _apply_rerank(self, query: str, docs: list) -> list:
        """统一重排入口。"""
        relevant_docs = docs
        if self.reranker_model and self.enable_reranker:
            filters_docs = self.reranker_model.rerank_documents(
                query=query,
                documents=relevant_docs,
                top_n=self.rerank_top_n,
                score_threshold=self.rerank_score_threshold
            )
            if filters_docs:
                relevant_docs = filters_docs
                logger.info(f"重排后提取 {len(relevant_docs)} 个相关文本块")
            else:
                relevant_docs = relevant_docs[:self.rerank_top_n]
                logger.info(f"重排后未筛选到文档，提取检索的 {len(relevant_docs)} 个相关文本块")
        return relevant_docs

    @staticmethod
    def _dedup_docs(docs: list) -> list:
        """按文档内容去重，避免上下文重复。"""
        seen = set()
        unique_docs = []
        for doc in docs:
            key = doc.page_content.strip()
            if key and key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs

    def _build_prompt(self, question: str, docs: list) -> str:
        """构造问答提示词。"""
        context_text = "\n\n".join([doc.page_content for doc in docs])
        system_prompt = """
        你是基于文档的问答助手，仅使用以下提供的文档片段（Context）回答问题。
        如果文档中没有相关信息，直接说“根据提供的文档，无法回答该问题”，不要编造内容。
        回答需简洁、准确，结合历史对话（History）理解上下文，每一次回答要重新审视当前提供的内容，不要只是简单重复历史回答。

        Context:
        {context_text}

        Current Question: {question}

        Answer:
        """
        return system_prompt.format(context_text=context_text, question=question)

    def _generate_answer_once(self, question: str, docs: list, history_messages: list) -> str:
        """基于给定文档上下文生成一次完整答案（非流式，用于双路对比）。"""
        final_prompt = self._build_prompt(question, docs)
        combine_contexts = list(history_messages)
        combine_contexts.append(HumanMessage(content=final_prompt))
        response = self.llm.invoke(combine_contexts)
        return (response.content or "").strip()

    def _pick_better_answer(self, question: str, raw_answer: str, concept_answer: str) -> str:
        """让LLM在两份候选答案中择优。"""
        judge_prompt = f"""
请在候选答案A和B中选择一个更好的最终答案，标准：
1) 忠实于文档，不编造；
2) 回答更完整、清晰；
3) 与问题更相关。

你必须只输出一个字母：A 或 B。
不要输出其他任何内容。

用户问题：{question}

候选答案A：
{raw_answer}

候选答案B：
{concept_answer}
"""
        try:
            selected = self.llm.invoke(judge_prompt)
            selected_text = (selected.content or "").strip().upper()
            choice_match = re.search(r"\b([AB])\b", selected_text)
            if choice_match:
                return raw_answer if choice_match.group(1) == "A" else concept_answer

            # 兼容模型返回“A.”、“答案A”等非严格格式
            if selected_text.startswith("A"):
                return raw_answer
            if selected_text.startswith("B"):
                return concept_answer

            logger.warning(f"答案择优输出非预期格式，回退概念增强答案：{selected_text}")
            return concept_answer or raw_answer
        except Exception as e:
            logger.warning(f"答案择优失败，默认返回概念增强答案：{str(e)}")
            return concept_answer or raw_answer

    def get_answer_stream(self, question: str) -> Generator[str, None, None]:
        """
        基于RAG技术生成问题答案，实现流式输出，逐块产生回答内容。
        核心流程：检索相关文档片段 →结合对话历史 → 拼接提示词 → 调用LLM生成答案。
        Args:
            question: 用户当前的问题（字符串类型，非空）
        Returns:
            生成的答案字符串；若发生错误，返回错误提示；若未上传文档，返回引导提示
        """
        # 重置当前流式回答
        self.current_stream_answer = ""

        # -------------------------- 1. 检查向量数据库是否初始化（是否已上传文档） --------------------------
        if not self.vectordb:
            yield "请先上传并处理文档，才能进行问答哦~"
            return
        if not question or not isinstance(question, str) or question.strip() == "":
            yield "请输入有效的问题内容~"
            return

        # -------------------------- 2. 对话历史记忆加载（适配长对话） --------------------------
        # 上下文管理
        combine_contexts = []
        # 加载对话历史记忆，设置窗口大小 k=50（保留最近100轮对话,滑动窗口机制），在维持对话连贯性的同时，避免历史记录过长导致的 Token 超限问题。
        for msg in self.memory.load_memory_variables({})["chat_history"]:
            combine_contexts.append(msg)

        # ------------------------ 3. 文档检索：原查询与概念增强查询 ------------------------
        raw_docs = self._apply_rerank(question, self._retrieve_docs(question))

        concept_docs = []
        concepts = []
        if self.enable_concept_expansion:
            concepts = self._extract_concepts(question)
            if concepts:
                expanded_query = f"{question}\n相关概念：{'、'.join(concepts)}"
                concept_docs = self._apply_rerank(expanded_query, self._retrieve_docs(expanded_query))
                logger.info(f"概念抽取：{concepts}")

        # ------------------------ 4. 双路策略：择优 or 合并 ------------------------
        if self.enable_concept_expansion and self.compare_with_raw_query and concept_docs:
            try:
                raw_answer = self._generate_answer_once(question, raw_docs, combine_contexts)
                concept_answer = self._generate_answer_once(question, concept_docs, combine_contexts)
                best_answer = self._pick_better_answer(question, raw_answer, concept_answer)

                for i in range(0, len(best_answer), 20):
                    chunk_text = best_answer[i:i + 20]
                    yield chunk_text
                    self.current_stream_answer += chunk_text

                self.memory.save_context(
                    inputs={"input": question},
                    outputs={"answer": self.current_stream_answer}
                )
                return
            except Exception as e:
                logger.warning(f"双路择优生成失败，降级为常规流式：{str(e)}")

        final_docs = raw_docs
        if concept_docs:
            final_docs = self._dedup_docs(raw_docs + concept_docs)

        # -------------------------- 5. 提示词构建：拼接完整提示词 --------------------------
        final_prompt = self._build_prompt(question, final_docs)
        combine_contexts.append(HumanMessage(content=final_prompt))

        logger.info(f"combine_contexts:{combine_contexts}")

        # -------------------------- 6. 流式调用LLM：逐块生成并返回答案 --------------------------
        try:
            # 流式调用LLM：llm.stream()返回生成器，逐块获取LLM输出（而非等待完整答案）
            for chunk in self.llm.stream(combine_contexts):
                # logger.info(f"chunk:{chunk}")
                # 实时返回输出内容
                if chunk.content:
                    yield chunk.content
                    self.current_stream_answer += chunk.content

            # 完整答案生成后，更新对话记忆：将本次问答（问题+完整答案）存入记忆，供下一轮对话复用
            self.memory.save_context(
                inputs={"input": question},
                outputs={"answer": self.current_stream_answer}
            )

            logger.info(f"self.memory.save_context:{self.memory.model_dump_json()}")

        except Exception as e:
            logger.error(f"错误：答案生成失败：{str(e)}")
            yield "抱歉，处理问题时发生错误，请稍后再试~"

    def get_answer(self, question: str) -> Optional[str]:
        """
        非流式问答方法，基于流式方法收集完整结果
        """
        full_answer = []
        for chunk in self.get_answer_stream(question):
            full_answer.append(chunk)
        return ''.join(full_answer)

    def clear_database(self) -> bool:
        """清空向量数据库"""
        try:
            if self.vectordb:
                # self.vectordb.delete_collection()
                # self.vectordb = None
                self.vectordb.reset_collection()

            # 清除记忆
            self.memory.clear()
            return True
        except Exception as e:
            logger.error(f"错误：数据库清空失败：{str(e)}")
            return False