

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# 添加模块路径，由于导入的llm模块位于当前文件main.py的上上级目录。否则会报找不到module异常
# module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
# if module_path not in sys.path:
#     sys.path.append(module_path)

# 将当前脚本所在目录加入Python搜索路径（确保能找到services目录下的RAGService）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from services.rag_service_stream import RAGService

# 加载环境变量
load_dotenv()

# 配置页面标题、图标、布局
st.set_page_config(
    # page_title="RAG知识问答助手",
    page_title="RAG知识问答",
    page_icon=":robot:",
    layout="wide"
)

# ===== UI 美化：统一蓝色主题 + 聊天气泡 + 控制面板分组 =====
st.markdown(
    """
<style>
/* 全局字体与背景微调 */
html, body, [class*="css"]  { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Arial; }
body {
  background: #F8FAFC;
}

/* 主色体系：青绿色（按示意图） */
:root {
  --brand: #0F7773;
  --brand-soft: #EAF7F6;
  --bg: #F4FBFA;
  --card: #FFFFFF;
  --border: #D7ECEA;
  --text: #0F2C2A;
  --radius: 12px;
  --shadow: 0 4px 12px rgba(15, 119, 115, 0.08);
}

/* 页面布局留白 */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  padding-left: 24px;
  padding-right: 24px;
}

/* 左侧栏宽度控制 */
section[data-testid="stSidebar"] {
  width: 312px !important;
  min-width: 312px !important;
}

/* 顶部说明卡片 */
.hero-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 12px;
  box-shadow: var(--shadow);
}
.hero-title {
  font-size: 1.18rem;
  font-weight: 700;
  color: #0F172A;
  margin-bottom: 6px;
}
.hero-subtitle {
  color: #2C5A58;
  font-size: 0.93rem;
  margin-bottom: 0;
}

.guide-card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}
.guide-title {
  color: #174543;
  font-size: 0.92rem;
  font-weight: 700;
  margin-bottom: 8px;
}
.guide-list {
  margin: 0;
  padding-left: 18px;
  color: #2F5B59;
  font-size: 0.88rem;
  line-height: 1.55;
}

/* 侧边栏整体背景 */
section[data-testid="stSidebar"] > div {
  background: var(--bg);
}

/* 侧边栏卡片容器：我们用 st.container 包起来时会更明显 */
.sidebar-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}

.section-title {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
}

/* 按钮：主按钮蓝色渐变，圆角 */
.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(15, 119, 115, 0.28) !important;
  background: linear-gradient(180deg, #15908A 0%, #0F7773 100%) !important;
  color: white !important;
  font-weight: 600 !important;
  box-shadow: 0 6px 16px rgba(15, 119, 115, 0.22) !important;
}
.stButton > button:hover { transform: translateY(-1px); }

/* 次按钮（secondary）更干净 */
.stButton > button[kind="secondary"] {
  background: white !important;
  color: #174543 !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}

/* 输入框、上传框、选择框统一圆角 */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  border-radius: 12px !important;
  border-color: #CFE5E3 !important;
}

/* 滑块样式 */
div[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--brand); }
div[data-testid="stSlider"] [role="slider"] {
  background: var(--brand) !important;
  border-color: var(--brand) !important;
}

/* 上传区域：虚线边框 + 居中提示感 */
div[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed #8CCFCC !important;
  border-radius: 12px !important;
  background: #F2FBFA !important;
}

/* Chat 气泡 */
div[data-testid="stChatMessage"] {
  border-radius: var(--radius);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  padding: 10px 14px;
  margin-bottom: 10px;
}

.user-bubble {
  background: var(--brand-soft);
  border: 1px solid #BFE3E1;
  border-radius: 12px;
  padding: 10px 12px;
}

.assistant-bubble {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
}
</style>
    """,
    unsafe_allow_html=True,
)

# 初始化Streamlit会话状态（跨刷新保存数据，避免页面刷新后数据丢失）
def initialize_app():
    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    # 用于重置文件上传框状态的会话变量
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    # 初始化RAG配置参数
    if "retrieve_k" not in st.session_state:
        st.session_state.retrieve_k = 6  # 默认检索文档数量
    if "enable_reranker" not in st.session_state:
        st.session_state.enable_reranker = False  # 默认开启重排
    if "rerank_top_n" not in st.session_state:
        st.session_state.rerank_top_n = 5  # 重排后保留文档数量
    if "enable_concept_expansion" not in st.session_state:
        st.session_state.enable_concept_expansion = False  # 默认关闭概念抽取增强检索
    if "concept_count" not in st.session_state:
        st.session_state.concept_count = 3  # 默认抽取概念数量
    if "compare_with_raw_query" not in st.session_state:
        st.session_state.compare_with_raw_query = False  # 默认关闭双路择优

    # 初始化RAG核心服务（封装了文档处理、向量存储、流式问答的核心逻辑）
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(
            retrieve_k=st.session_state.retrieve_k,
            enable_reranker=st.session_state.enable_reranker,
            enable_concept_expansion=st.session_state.enable_concept_expansion,
            concept_count=st.session_state.concept_count,
            compare_with_raw_query=st.session_state.compare_with_raw_query,
            rerank_top_n=st.session_state.rerank_top_n,
            # 重排模型配置：用于对检索结果进行语义重排序.默认使用BAAI的bge-reranker-v2-m3（中文效果性价比较高）。
            # 这里加载本地路径模型（需手动下载模型文件到指定路径）
            # a.模型文件获取地址：https://modelscope.cn/models/BAAI/bge-reranker-v2-m3
            # b.需下载文件：config.json、model.safetensors、special_tokens_map.json、tokenizer.json、tokenizer_config.json
            model_name_or_path="/mnt/nvme0n1/zcl/RAG/simple_rag_assistant/data/models_reranker_data/BAAI/bge-reranker-v2-m3"  # 指定重排模型本地存储路径
        )


initialize_app()

# 定义侧边栏区域
with st.sidebar:
    st.subheader("RAG知识问答")
    st.caption("配置你的检索、重排与查询增强策略")

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">检索与重排配置</div>', unsafe_allow_html=True)

    # RAG检索配置区域
    # 1. 检索数量控制
    retrieve_k = st.slider(
        "初始检索文档数量 (retrieve_k)",
        min_value=1, max_value=10, value=st.session_state.retrieve_k, step=1,
        help="从向量库中初始检索的文档数量，数量越多覆盖范围越广，但可能引入噪音"
    )

    # 2. 重排功能开关
    enable_reranker = st.toggle(
        "开启检索结果重排",
        value=st.session_state.enable_reranker,
        help="开启后会对检索到的文档进行语义重排序，提升回答质量，但会增加响应时间"
    )

    # 3. 重排后保留数量（仅在开启重排时可配置）
    rerank_top_n = st.slider(
        "重排后保留文档数量 (rerank_top_n)",
        min_value=1, max_value=8, value=st.session_state.rerank_top_n, step=1,
        help="重排后最终保留的文档数量，需小于等于初始检索数量",
        disabled=not enable_reranker  # 关闭重排时禁用该参数
    )

    # 限制rerank_top_n不超过retrieve_k
    if rerank_top_n > retrieve_k:
        rerank_top_n = retrieve_k
        st.warning(f"重排保留数量自动调整为 {retrieve_k}（不超过初始检索数量）")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">查询增强配置</div>', unsafe_allow_html=True)
    enable_concept_expansion = st.toggle(
        "开启概念抽取增强检索",
        value=st.session_state.enable_concept_expansion,
        help="先用LLM抽取查询中的关键概念，再与原查询联合检索，提高召回覆盖"
    )
    concept_count = st.slider(
        "概念抽取数量",
        min_value=1, max_value=8, value=st.session_state.concept_count, step=1,
        disabled=not enable_concept_expansion,
        help="控制LLM从问题中抽取的概念个数"
    )
    compare_with_raw_query = st.toggle(
        "双路对比择优（有概念 vs 无概念）",
        value=st.session_state.compare_with_raw_query,
        disabled=not enable_concept_expansion,
        help="开启后将分别生成两份答案，并由LLM自动选择更优答案输出"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 应用配置", use_container_width=True, type="primary"):
        # 更新会话状态
        st.session_state.enable_reranker = enable_reranker
        st.session_state.retrieve_k = retrieve_k
        st.session_state.rerank_top_n = rerank_top_n
        st.session_state.enable_concept_expansion = enable_concept_expansion
        st.session_state.concept_count = concept_count
        st.session_state.compare_with_raw_query = compare_with_raw_query

        # 更新RAGService的配置
        st.session_state.rag_service.enable_reranker = enable_reranker
        st.session_state.rag_service.retrieve_k = retrieve_k
        st.session_state.rag_service.rerank_top_n = rerank_top_n
        st.session_state.rag_service.enable_concept_expansion = enable_concept_expansion
        st.session_state.rag_service.concept_count = concept_count
        st.session_state.rag_service.compare_with_raw_query = compare_with_raw_query

        st.success("配置已更新生效！")

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">文档管理</div>', unsafe_allow_html=True)
    # 1. 多文件上传（支持PDF/DOCX/TXT/MD，与RAGService支持的格式一致）
    uploaded_files = st.file_uploader(
        "📂 上传课程资料\n支持 PDF / DOCX / TXT / MD\n拖拽文件到此处或点击上传",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.upload_key}"  # 动态生成key
    )
    # 处理上传的文件：调用RAGService的process_document方法，完成“解析→分块→向量化→入库”
    if uploaded_files:
        with st.spinner("正在处理文档..."):
            for file in uploaded_files:
                st.session_state.rag_service.process_document(file)
            st.success(f"已成功处理 {len(uploaded_files)} 个文档")
            # 处理完成后重置上传框：通过改变key值实现
            st.session_state.upload_key += 1

    # 2. 清空知识库（删除向量库数据+清空聊天历史，重置整个问答环境）
    if st.button("清空知识库", type="secondary", use_container_width=True):
        with st.spinner("正在清空知识库..."):
            # 清空向量存储
            st.session_state.rag_service.clear_database()
            # 清空聊天历史
            st.session_state.history = []
            st.success("知识库已成功清空")
    st.markdown('</div>', unsafe_allow_html=True)

# 主界面 - 聊天区域
st.markdown(
    f"""
<div class="hero-card">
  <div class="hero-title">课程知识智能问答系统</div>
  <div class="hero-subtitle">上传资料后即可进行知识问答，系统支持检索增强、重排与查询增强策略。</div>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.history:
    st.markdown(
        """
<div class="guide-card">
  <div class="guide-title">欢迎使用 👋 你可以试试这些问题：</div>
  <ul class="guide-list">
    <li>请总结这份资料的核心内容。</li>
    <li>这份课程学习资料主要是讲了什么？</li>
    <li>根据文档，帮我整理一个考试复习提纲。</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

# 1. 展示聊天历史（遍历session_state.history，按角色显示消息）
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"<div class='user-bubble'> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'> {message['content']}</div>", unsafe_allow_html=True)

# 2. 处理用户输入
user_input = st.chat_input("请问有什么可以帮助您？")
if user_input:
    # 步骤1：将用户消息添加到会话历史
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'> {user_input}</div>", unsafe_allow_html=True)

    # 步骤2：调用RAG服务生成流式回答，并显示
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # RAG回答，非流式：大模型完整输出后才展示出来
            # full_answer = rag_service.get_answer(user_input)
            # st.markdown(full_answer)

            # RAG回答，流式输出
            full_answer = ""  # 用于存储完整的回复内容
            # 调用RAGService的get_answer_stream（流式方法），用st.write_stream实现边生成边显示
            for chunk in st.write_stream(st.session_state.rag_service.get_answer_stream(user_input)):
                full_answer += chunk

            # 步骤3：将完整的助手回答添加到会话历史，供下次刷新时展示
            st.session_state.history.append({"role": "assistant", "content": full_answer})
            st.rerun()

