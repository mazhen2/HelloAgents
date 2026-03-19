"""
LangChain 自定义会话管理与 Retriever 示例
包含：会话历史管理、自定义Retriever、完整 Conversational RAG 实现

本文件演示 LangChain 中会话管理和检索器的核心用法：
1. ChatSessionManager: 手动管理多个会话的历史记录
2. RunnableWithMessageHistory: 自动管理会话历史的 Runnable
3. AnimalRetriever: 基于关键词匹配的自定义检索器
4. build_conversational_rag_chain: 完整的 Conversational RAG 实现

这些组件使得构建多轮对话应用和上下文感知的 RAG 系统成为可能。

依赖安装：
pip install faiss-cpu langchain-text-splitters

==========================================
RAG 核心概念详解
==========================================

【什么是 RAG？】
RAG = Retrieval-Augmented Generation（检索增强生成）
是一种将外部知识检索与 LLM 生成能力相结合的架构模式。

核心流程：
用户问题 → 检索相关文档 → 将文档作为上下文 → LLM 生成答案

【为什么需要 RAG？】
1. 解决 LLM 知识过时问题 - LLM 训练数据有截止日期
2. 减少幻觉 - 基于实际文档生成，减少胡说八道
3. 可追溯性 - 答案来源于具体文档，可验证
4. 私有知识 - 让 LLM 理解企业/个人私有文档

【普通 RAG vs Conversational RAG】
普通 RAG：
- 每次都是独立的问题
- 无法理解"它"、"那个"等指代词
- 无法处理省略句

Conversational RAG（对话式 RAG）：
- 维护完整的对话历史
- 能理解代词和上下文
- 支持多轮追问
- 例如：
  用户: "什么是 LangChain？"
  AI: "LangChain 是一个..."
  用户: "它的核心组件有哪些？"  ← "它"需要结合上文理解

【Conversational RAG 的关键技术】
1. History-Aware Retriever（历史感知检索器）
   - 将带指代词的问题改写为独立问题
   - 例如："它有哪些组件？" → "LangChain 有哪些核心组件？"

2. Session Management（会话状态管理）
   - 为每个用户/对话维护独立的历史
   - 支持上下文累积

3. Context Rewriting（上下文重写）
   - 使用 LLM 将问题改写为独立查询
   - 这是 History-Aware Retriever 的核心

【本文件 RAG 实现详解】
我们使用 langchain-core 基础组件构建 RAG，不依赖 langchain.chains 模块：

1. TfidfEmbeddings: 简单的 TF-IDF 嵌入实现（演示用）
   - 实际生产中应使用 OpenAI、DashScope 等更强大的 embedding

2. build_conversational_rag_chain() 函数实现了完整流程：
   - 文档分割：将长文档切分成小块
   - 向量存储：使用 FAISS 构建向量索引
   - 基础检索：基于向量相似度检索相关文档
   - 历史感知检索：用 LLM 改写问题后再检索
   - 答案生成：基于检索结果和历史生成答案
   - 会话包装：使用 RunnableWithMessageHistory 管理历史
"""

import os
from typing import List, Dict, Any

# 从 .env 文件加载环境变量
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# 消息历史存储
from langchain_community.chat_message_histories import ChatMessageHistory

# 文档分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量存储和 Embedding
from langchain_community.vectorstores import FAISS

# 用于简单 embedding
from sklearn.feature_extraction.text import TfidfVectorizer

# OpenAI 兼容的 LLM 接口
from langchain_openai import ChatOpenAI

# 加载 .env 文件
load_dotenv()

# =============================================================================
# 模型初始化 - 从环境变量读取配置
# =============================================================================

# 模型名称：支持自定义，默认为 qwen3.5-flash
model_name = os.getenv("LLM_MODEL_ID", "qwen3.5-flash")

# API 密钥：用于身份验证
api_key = os.getenv("LLM_API_KEY")

# API 基础 URL：LLM 服务的端点地址
base_url = os.getenv("LLM_BASE_URL")

# 请求超时时间（秒）
timeout = int(os.getenv("LLM_TIMEOUT", "60"))

# Embedding 配置（使用 TF-IDF 本地 embedding）
# 简化实现，无需外部 embedding API

# =============================================================================
# RAG 配置参数详解
# =============================================================================
# 以下参数控制 RAG 系统的行为，可以根据实际需求调整

# top_k: 检索返回的顶部文档数量
# - 默认为 2 个最相关结果
# - 值越大，检索范围越广，但可能引入更多噪声
# - 值越小，结果越精准，但可能遗漏重要信息
# - 建议范围：1-10，根据文档长度和质量调整
top_k = int(os.getenv("RAG_TOP_K", "2"))

# chunk_size: 文本分块大小（每块字符数）
# - 默认为 500 字符
# - 值太大：每个块包含信息多，但可能包含无关内容
# - 值太小：信息精炼，但可能丢失上下文
# - 建议范围：200-1000，根据内容类型调整
# - 中文内容建议 300-600，英文建议 500-1000
chunk_size = int(os.getenv("CHUNK_SIZE", "500"))

# chunk_overlap: 分块重叠大小（相邻块重叠的字符数）
# - 默认为 50 字符
# - 作用：保持上下文连贯性，避免重要信息被切分到不同块
# - 值太大：冗余信息多，检索效率低
# - 值太小：可能丢失块边界附近的上下文
# - 建议为 chunk_size 的 10%-20%
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))


# =============================================================================
# 模型初始化 - 从环境变量读取配置
# =============================================================================
# ChatOpenAI 是 LangChain 提供的 OpenAI 兼容 LLM 接口
# 支持多种 OpenAI 兼容的 API，如 DashScope、Ollama、自建模型等

# 初始化全局 LLM 模型实例
# 参数说明：
# - model: 模型名称，支持自定义配置
# - api_key: API 密钥，用于身份验证（部分服务商需要）
# - base_url: API 基础地址，指向 LLM 服务端点
# - timeout: 请求超时时间，防止长时间等待响应
model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    timeout=timeout,
)


# =============================================================================
# 一、自定义会话管理
# =============================================================================
# 本节展示如何手动管理会话历史，适用于需要细粒度控制的场景
#
# 【会话管理的必要性】
# 在构建聊天机器人或对话系统时，需要维护对话上下文：
# - 让 AI 记住之前的对话内容
# - 支持多轮对话中的指代消解（"它"、"那个"）
# - 为不同用户/对话维护独立的上下文
#
# 【会话管理的两种方式】
# 1. ChatSessionManager（手动管理）：
#    - 完全控制会话历史的存储和读取
#    - 需要手动调用 add_user_message/add_ai_message
#    - 适合需要特殊处理逻辑的场景
#
# 2. RunnableWithMessageHistory（自动管理）：
#    - LangChain 提供的自动封装
#    - 自动注入历史到提示词
#    - 自动保存新消息到历史
#    - 适合标准的多轮对话场景


class ChatSessionManager:
    """
    会话管理器 - 管理多个会话的历史记录
    
    【什么是会话管理？】
    在多轮对话系统中，需要维护每个用户/对话的历史消息。
    会话管理器负责：
    1. 创建和存储会话历史
    2. 根据 session_id 获取对应历史
    3. 管理会话的生命周期（创建、查询、清除）
    
    【为什么需要会话管理器？】
    1. 上下文记忆：让 AI 记住之前的对话内容
    2. 多用户隔离：不同用户有独立的会话空间
    3. 上下文限制：AI 有最大上下文长度限制，需要选择性保留历史
    
    【存储方式】
    - 当前实现：内存字典（仅演示用）
    - 生产环境建议：
      * Redis：高性能，适合高并发场景
      * 数据库：MySQL/PostgreSQL，适合需要持久化的场景
      * 文件系统：简单，适合单机场景
    
    【与 RunnableWithMessageHistory 的关系】
    - ChatSessionManager：底层的会话存储实现
    - RunnableWithMessageHistory：上层封装，自动调用会话管理器
    - 可以直接使用 ChatSessionManager 进行更灵活的控制
    
    Attributes:
        store: 存储会话 ID 到消息历史的映射字典
            格式：{session_id: ChatMessageHistory}
    """

    def __init__(self):
        """
        初始化会话管理器
        
        创建一个空的消息历史存储字典。
        session_id → ChatMessageHistory 的映射
        """
        self.store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        获取指定会话的历史记录
        
        【重要】这是一个惰性创建的方法：
        如果会话不存在，会自动创建一个新的空会话。
        这简化了使用逻辑，无需预先检查会话是否存在。
        
        Args:
            session_id: 会话的唯一标识符
                通常由业务逻辑生成，如：
                - 用户ID：user_123
                - 用户ID+日期：user_123_20240101
                - UUID：550e8400-e29b-41d4-a716-446655440000
        
        Returns:
            ChatMessageHistory: 该会话的消息历史对象
                可以调用 add_user_message() 和 add_ai_message() 添加消息
        """
        if session_id not in self.store:
            # 会话不存在时，自动创建新的空会话
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_all_sessions(self) -> List[str]:
        """
        获取所有会话ID
        
        用于查看当前管理器中存储了多少会话。
        可用于监控、调试或清理过期会话。
        
        Returns:
            List[str]: 所有已存在的会话 ID 列表
        """
        return list(self.store.keys())

    def clear_session(self, session_id: str) -> None:
        """
        清除指定会话
        
        【使用场景】
        - 用户主动结束会话
        - 会话超时过期
        - 用户请求重置对话
        
        Args:
            session_id: 要清除的会话 ID
        """
        if session_id in self.store:
            del self.store[session_id]

    def print_session_history(self, session_id: str) -> None:
        """
        打印会话历史
        
        用于调试和查看会话内容。
        生产环境可能需要更友好的展示方式。
        
        Args:
            session_id: 会话 ID
        """
        if session_id not in self.store:
            print(f"会话 {session_id} 不存在")
            return

        messages = self.store[session_id].messages
        if not messages:
            print(f"会话 {session_id} 无历史记录")
            return

        # 遍历并打印每条消息
        for message in messages:
            # 根据消息类型选择前缀
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"
            print(f"{prefix}: {message.content}\n")


def demo_chat_session_manager():
    """
    会话管理器演示
    
    ==========================================
    ChatSessionManager 使用演示
    ==========================================
    
    展示如何使用 ChatSessionManager 管理多个独立会话。
    
    【核心概念】
    - 每个会话（session_id）都有独立的历史记录
    - 不同会话之间互不影响
    - 可以创建任意数量的会话
    
    【使用场景】
    - 多用户聊天机器人
    - 多会话测试
    - 需要隔离的对话场景
    """
    print("\n" + "=" * 50)
    print("1. 基础会话管理器")
    print("=" * 50)

    # =========================================================================
    # 【创建会话管理器实例】
    # =========================================================================
    # 会话管理器管理所有会话的历史
    # 内部使用字典存储：{session_id: ChatMessageHistory}
    
    manager = ChatSessionManager()

    # =========================================================================
    # 【添加消息到会话】
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 【会话1：中文对话】
    # -------------------------------------------------------------------------
    # 通过 get_session_history 获取会话历史
    # 如果会话不存在，会自动创建
    history1 = manager.get_session_history("session_1")
    
    # 添加用户消息和 AI 回复
    history1.add_user_message("你好")  # 用户说"你好"
    history1.add_ai_message("你好，有什么可以帮你的？")  # AI 回复

    # -------------------------------------------------------------------------
    # 【会话2：英文对话】
    # -------------------------------------------------------------------------
    # 不同会话有独立的历史
    # session_2 会自动创建（因为之前不存在）
    history2 = manager.get_session_history("session_2")
    history2.add_user_message("Hello")
    history2.add_ai_message("Hello, how can I help?")

    # =========================================================================
    # 【查询会话信息】
    # =========================================================================
    
    # 获取所有会话 ID
    all_sessions = manager.get_all_sessions()
    print(f"当前所有会话: {all_sessions}")

    # 打印指定会话的历史
    print("\nsession_1 历史:")
    manager.print_session_history("session_1")
    
    print("\nsession_2 历史:")
    manager.print_session_history("session_2")
    
    # =========================================================================
    # 【会话管理操作】
    # =========================================================================
    
    # 清除指定会话
    # 这会删除该会话的所有历史记录
    print("\n清除 session_2:")
    manager.clear_session("session_2")
    print(f"清除后剩余会话: {manager.get_all_sessions()}")
    
    # 尝试打印已清除的会话
    print("\n尝试打印已清除的 session_2:")
    manager.print_session_history("session_2")
    """
    会话管理器 - 管理多个会话的历史记录
    
    【什么是会话管理？】
    在多轮对话系统中，需要维护每个用户/对话的历史消息。
    会话管理器负责：
    1. 创建和存储会话历史
    2. 根据 session_id 获取对应历史
    3. 管理会话的生命周期（创建、查询、清除）
    
    【为什么需要会话管理器？】
    1. 上下文记忆：让 AI 记住之前的对话内容
    2. 多用户隔离：不同用户有独立的会话空间
    3. 上下文限制：AI 有最大上下文长度限制，需要选择性保留历史
    
    【存储方式】
    - 当前实现：内存字典（仅演示用）
    - 生产环境建议：
      * Redis：高性能，适合高并发场景
      * 数据库：MySQL/PostgreSQL，适合需要持久化的场景
      * 文件系统：简单，适合单机场景
    
    【与 RunnableWithMessageHistory 的关系】
    - ChatSessionManager：底层的会话存储实现
    - RunnableWithMessageHistory：上层封装，自动调用会话管理器
    - 可以直接使用 ChatSessionManager 进行更灵活的控制
    
    Attributes:
        store: 存储会话 ID 到消息历史的映射字典
            格式：{session_id: ChatMessageHistory}
    """

    def __init__(self):
        """
        初始化会话管理器
        
        创建一个空的消息历史存储字典。
        session_id → ChatMessageHistory 的映射
        """
        self.store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        获取指定会话的历史记录
        
        【重要】这是一个惰性创建的方法：
        如果会话不存在，会自动创建一个新的空会话。
        这简化了使用逻辑，无需预先检查会话是否存在。
        
        Args:
            session_id: 会话的唯一标识符
                通常由业务逻辑生成，如：
                - 用户ID：user_123
                - 用户ID+日期：user_123_20240101
                - UUID：550e8400-e29b-41d4-a716-446655440000
        
        Returns:
            ChatMessageHistory: 该会话的消息历史对象
                可以调用 add_user_message() 和 add_ai_message() 添加消息
        """
        if session_id not in self.store:
            # 会话不存在时，自动创建新的空会话
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_all_sessions(self) -> List[str]:
        """
        获取所有会话ID
        
        用于查看当前管理器中存储了多少会话。
        可用于监控、调试或清理过期会话。
        
        Returns:
            List[str]: 所有已存在的会话 ID 列表
        """
        return list(self.store.keys())

    def clear_session(self, session_id: str) -> None:
        """
        清除指定会话
        
        【使用场景】
        - 用户主动结束会话
        - 会话超时过期
        - 用户请求重置对话
        
        Args:
            session_id: 要清除的会话 ID
        """
        if session_id in self.store:
            del self.store[session_id]

    def print_session_history(self, session_id: str) -> None:
        """
        打印会话历史
        
        用于调试和查看会话内容。
        生产环境可能需要更友好的展示方式。
        
        Args:
            session_id: 会话 ID
        """
        if session_id not in self.store:
            print(f"会话 {session_id} 不存在")
            return

        messages = self.store[session_id].messages
        if not messages:
            print(f"会话 {session_id} 无历史记录")
            return

        # 遍历并打印每条消息
        for message in messages:
            # 根据消息类型选择前缀
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"
            print(f"{prefix}: {message.content}\n")


def demo_chat_session_manager():
    """
    会话管理器演示

    展示如何使用 ChatSessionManager 管理多个独立会话。
    每个会话都有独立的历史记录，互不影响。
    """
    print("\n" + "=" * 50)
    print("1. 基础会话管理器")
    print("=" * 50)

    # 创建会话管理器实例
    manager = ChatSessionManager()

    # 获取 session_1 的历史记录并添加消息
    history1 = manager.get_session_history("session_1")
    history1.add_user_message("你好")
    history1.add_ai_message("你好，有什么可以帮你的？")

    # 获取 session_2 的历史记录（独立会话）
    history2 = manager.get_session_history("session_2")
    history2.add_user_message("Hello")
    history2.add_ai_message("Hello, how can I help?")

    # 打印所有会话 ID
    print(f"所有会话: {manager.get_all_sessions()}")

    # 打印 session_1 的历史
    print("\nsession_1 历史:")
    manager.print_session_history("session_1")


def demo_runnable_with_message_history():
    """
    RunnableWithMessageHistory 使用示例
    
    ==========================================
    RunnableWithMessageHistory 详解
    ==========================================
    
    【什么是 RunnableWithMessageHistory？】
    LangChain 提供的高级封装类，用于给任意 Runnable（链）添加会话历史管理能力。
    
    【解决的问题】
    1. 自动注入历史：在调用链之前，自动将历史消息注入到输入中
    2. 自动保存消息：在调用链之后，自动将用户消息和 AI 输出保存到历史
    3. 多会话隔离：通过 session_id 区分不同用户的会话
    
    【工作原理图】
    
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. 调用前                                                    │
    │    session_manager.get_session_history(session_id)           │
    │    获取历史 → 注入到输入字典                                  │
    │    输入: {input: "问题"}                                     │
    │    变成: {input: "问题", chat_history: [消息1, 消息2, ...]}   │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 2. 执行链                                                    │
    │    prompt.format(chat_history=..., input=...)                 │
    │    → llm.invoke(...)                                        │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 3. 调用后                                                    │
    │    session_manager.add_user_message(input)                    │
    │    session_manager.add_ai_message(output)                    │
    │    自动保存到会话历史                                         │
    └──────────────────────────────────────────────────────────────┘
    
    【关键参数】
    - chain: 要包装的链（任意 Runnable）
    - get_session_history: 获取会话历史的函数
    - input_messages_key: 用户输入的键名（prompt 中的占位符）
    - history_messages_key: 历史消息在 prompt 中的占位符名
    - output_messages_key: AI 输出的键名（用于保存）
    
    【与 ChatSessionManager 的关系】
    RunnableWithMessageHistory 内部调用 ChatSessionManager，
    但封装了更高级的逻辑，如消息格式转换、自动保存等。
    """
    print("\n" + "=" * 50)
    print("2. RunnableWithMessageHistory")
    print("=" * 50)

    # =========================================================================
    # 【会话历史存储】
    # =========================================================================
    # 
    # 使用内存字典存储会话历史
    # Key: session_id
    # Value: ChatMessageHistory 对象
    
    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        """
        获取会话历史的回调函数
        
        【为什么需要这个函数？】
        RunnableWithMessageHistory 需要一个函数来获取会话历史。
        这个函数在每次调用时都会被调用。
        
        【函数签名要求】
        - 参数：session_id (str)
        - 返回值：ChatMessageHistory 对象
        
        【实现说明】
        - 如果会话不存在，创建一个新的空会话
        - 这是惰性初始化的模式
        
        Args:
            session_id: 会话 ID
        
        Returns:
            ChatMessageHistory: 该会话的消息历史
        """
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # =========================================================================
    # 【创建提示词模板】
    # =========================================================================
    #
    # 【MessagesPlaceholder 的作用】
    # 这是一个占位符，会被替换为实际的聊天历史
    # 通过 variable_name 与 RunnableWithMessageHistory 的 history_messages_key 对应
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),  # 系统指令
        MessagesPlaceholder(variable_name="chat_history"),  # ← 这里会被替换为历史
        ("human", "{input}"),  # 用户输入
    ])

    # 构建链：提示词模板 → 模型
    # 使用 | 管道操作符连接组件
    chain = prompt | model

    # =========================================================================
    # 【包装成带会话历史的链】
    # =========================================================================
    #
    # 【关键参数说明】
    # - chain: 要包装的链
    # - get_session_history: 上面定义的回调函数
    # - input_messages_key: "input" - 用户输入在字典中的键
    # - history_messages_key: "chat_history" - 对应 MessagesPlaceholder 的 variable_name
    
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # =========================================================================
    # 【测试调用】
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 【第一次调用】没有历史
    # -------------------------------------------------------------------------
    print("\n第一次调用（无历史）:")
    print("用户: 你好，我叫张三")
    
    # 通过 config 指定 session_id
    # config 是 LangChain 的标准配置格式
    # configurable 用于传递自定义参数（如 session_id）
    response1 = conversational_chain.invoke(
        {"input": "你好，我叫张三"},  # 输入字典
        config={"configurable": {"session_id": "abc123"}},  # 会话 ID
    )
    print(f"AI: {response1.content}")

    # -------------------------------------------------------------------------
    # 【第二次调用】有历史，能记住名字
    # -------------------------------------------------------------------------
    print("\n第二次调用（有历史）:")
    print("用户: 我叫什么名字？")
    
    # 使用相同的 session_id
    # RunnableWithMessageHistory 会自动：
    # 1. 获取 abc123 的历史（包含上一轮对话）
    # 2. 将历史注入到提示词中
    # 3. 调用链
    # 4. 保存新消息到历史
    
    response2 = conversational_chain.invoke(
        {"input": "我叫什么名字？"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(f"AI: {response2.content}")

    # -------------------------------------------------------------------------
    # 【查看会话历史】
    # -------------------------------------------------------------------------
    print("\n会话历史内容:")
    for message in store["abc123"].messages:
        prefix = "AI" if isinstance(message, AIMessage) else "User"
        print(f"  {prefix}: {message.content[:50]}...")


# =============================================================================
# 二、自定义 Retriever
# =============================================================================
# 本节展示如何创建自定义检索器
#
# 【Retriever 接口简介】
# Retriever 是 LangChain 中用于检索文档的接口。
# 它只需要实现 _get_relevant_documents 方法（或其异步版本）。
#
# 【为什么需要自定义 Retriever？】
# 1. 简单关键词匹配：不想用向量数据库的场景
# 2. 混合检索：结合多个数据源
# 3. 业务逻辑：特殊的排序或过滤规则
# 4. 原型验证：快速验证检索思路
#
# 【BaseRetriever 提供了什么？】
# 继承 BaseRetriever 后自动获得：
# - Runnable 接口：可直接用于 Chain
# - async/await 支持：异步调用能力
# - Callback 集成：与 LangChain 回调系统集成
# - invoke/batch/ainvoke 等方法：标准调用方式


class AnimalRetriever(BaseRetriever):
    """
    动物检索器 - 基于关键词匹配的自定义检索器
    
    【这是一个演示用的简单检索器】
    实际应用中应使用向量相似度搜索等更复杂的检索逻辑。
    
    【设计目的】
    演示如何继承 BaseRetriever 创建自定义检索器。
    使用简单的关键词匹配（包含关系）来找到相关文档。
    
    【关键词匹配 vs 向量检索】
    关键词匹配（当前实现）：
    - 优点：简单直观，无需模型
    - 缺点：不理解语义，"狗"搜不到"犬"
    
    向量检索（推荐）：
    - 优点：语义理解，"狗"能搜到"犬"
    - 缺点：需要 embedding 模型
    
    【继承 BaseRetriever 的好处】
    1. 自动实现 Runnable 接口：可以直接用于 chain | retriever
    2. 自动支持异步：提供 _aget_relevant_documents 的默认实现
    3. 自动集成回调：run_manager 可以追踪执行过程
    4. 自动处理参数：search_kwargs 等标准参数
    
    Attributes:
        docs: 文档列表，用于检索的文档集合
        k: 返回的最大文档数量限制
    
    Example:
        >>> docs = [Document(page_content="狗很忠诚"), Document(page_content="猫很独立")]
        >>> retriever = AnimalRetriever(docs=docs, k=1)
        >>> results = retriever.invoke("宠物")
        >>> # 返回包含"宠物"的文档
    """

    # 使用 Pydantic 风格声明字段类型
    # 这让 BaseRetriever 知道如何验证和处理这些属性
    docs: List[Document]  # 文档列表
    k: int  # 返回结果数量上限

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        同步获取与查询相关的文档（核心方法）
        
        【方法签名要点】
        - query: 用户查询文本
        - run_manager: 回调管理器，用于：
          * 追踪执行过程
          * 记录日志
          * 传递上下文信息
        
        【实现逻辑】
        1. 遍历所有文档
        2. 检查文档内容是否包含查询词（不区分大小写）
        3. 收集匹配结果，直到达到 k 个
        4. 返回匹配的文档列表
        
        【为什么不直接返回所有匹配？】
        - 避免返回过多结果，保持输出可控
        - 减少后续处理的开销
        - 符合 top-k 检索的常见设计
        
        Args:
            query: 用户查询，如 "宠物"、"冬眠"
            run_manager: 回调管理器，用于追踪执行
        
        Returns:
            List[Document]: 匹配的文档列表，最多 k 个
        """
        matching_documents = []
        
        for document in self.docs:
            # 达到返回数量限制时停止
            # 这是常见的优化：不需要遍历所有文档
            if len(matching_documents) >= self.k:
                break
            
            # 简单的关键词匹配（不区分大小写）
            # 使用 lower() 将查询和文档都转为小写进行匹配
            # 这样 "狗" 能匹配到 "狗是人类的伴侣"
            # 但 "犬" 不能匹配到 "狗是..."
            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        
        return matching_documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        异步获取与查询相关的文档
        
        【为什么需要异步版本？】
        某些检索操作可能是 I/O 密集型的，如：
        - 数据库查询
        - API 调用
        - 网络请求
        
        异步版本可以让这些操作不阻塞主线程。
        
        【默认实现】
        如果不需要异步支持，可以不实现这个方法。
        BaseRetriever 提供了默认的异步实现（通过 asyncio.to_thread）。
        但显式实现可以提供更好的性能和更精确的控制。
        
        【当前实现】
        这里的异步版本与同步版本逻辑相同，
        因为当前的关键词匹配是 CPU 计算，无需真正的异步。
        
        Args:
            query: 用户查询
            run_manager: 异步回调管理器
        
        Returns:
            List[Document]: 匹配的文档列表
        """
        matching_documents = []
        
        for document in self.docs:
            if len(matching_documents) >= self.k:
                break
            
            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        
        return matching_documents


def demo_custom_retriever():
    """
    自定义 Retriever 示例
    
    ==========================================
    检索器使用演示
    ==========================================
    
    展示如何创建和使用自定义检索器。
    
    【使用方式】
    1. 继承 BaseRetriever
    2. 实现 _get_relevant_documents 方法
    3. 使用 invoke() 调用检索器
    
    【与 VectorStore Retriever 的区别】
    - VectorStore Retriever：由 vectorstore.as_retriever() 创建
    - Custom Retriever：继承 BaseRetriever 自定义实现
    - 两者都实现相同的接口，可以互换使用
    """
    print("\n" + "=" * 50)
    print("3. 自定义 Retriever")
    print("=" * 50)

    # =========================================================================
    # 【创建测试文档】
    # =========================================================================
    #
    # Document 是 LangChain 的标准文档格式
    # 包含：
    # - page_content: 文档内容（字符串）
    # - metadata: 元数据（字典，可选）
    
    documents = [
        Document(page_content="狗是人类的伴侣，以其忠诚著称。"),
        Document(page_content="猫是人类的伴侣，通常喜欢独立的空间。"),
        Document(page_content="兔子是宠物，通常喜欢吃胡萝卜。"),
        Document(page_content="鱼是宠物，通常生活在水里。"),
        Document(page_content="鸟是动物，通常会唱歌。"),
        Document(page_content="熊是动物，通常会冬眠。"),
    ]
    
    print(f"创建了 {len(documents)} 个文档用于检索演示")

    # =========================================================================
    # 【初始化检索器】
    # =========================================================================
    #
    # AnimalRetriever 的参数：
    # - docs: 要检索的文档列表
    # - k: 返回的最大文档数量
    #
    # 【k 值的选择】
    # - k=1: 只返回最相关的一个
    # - k=2: 返回两个（演示用）
    # - k=3-5: 常见于 RAG 系统
    
    retriever = AnimalRetriever(docs=documents, k=2)
    print(f"检索器初始化完成，每次最多返回 {retriever.k} 个文档")

    # =========================================================================
    # 【测试检索】
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 【测试1】检索"宠物"
    # -------------------------------------------------------------------------
    print("\n检索 '宠物':")
    
    # 使用 invoke 方法调用检索器
    # 这是 Runnable 接口的标准方法
    # 返回值是匹配文档的列表
    results = retriever.invoke("宠物")
    
    print(f"找到 {len(results)} 个相关文档:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    # -------------------------------------------------------------------------
    # 【测试2】检索"冬眠"
    # -------------------------------------------------------------------------
    print("\n检索 '冬眠':")
    
    results = retriever.invoke("冬眠")
    
    print(f"找到 {len(results)} 个相关文档:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    # -------------------------------------------------------------------------
    # 【测试3】检索不存在的关键词
    # -------------------------------------------------------------------------
    print("\n检索 '火箭':")
    
    results = retriever.invoke("火箭")
    
    print(f"找到 {len(results)} 个相关文档（预期为0）:")
    if not results:
        print("  （无匹配结果）")
    else:
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")


# =============================================================================
# 三、完整 Conversational RAG 实现 - 核心组件
# =============================================================================
# 本节展示如何构建结合会话历史的 RAG 系统
# 
# RAG 系统的核心组件：
# 1. Embeddings（嵌入模型）：将文本转换为向量
# 2. Vector Store（向量存储）：存储文档向量，支持相似度检索
# 3. Retriever（检索器）：根据查询检索相关文档
# 4. Chain（链）：组合各个组件形成完整流程


class TfidfEmbeddings:
    """
    简单的 TF-IDF Embedding 实现
    
    【什么是 Embedding？】
    Embedding（嵌入）是将文本转换为数值向量的技术。
    语义相近的文本在向量空间中距离更近。
    例如："狗"和"宠物"的向量距离比"狗"和"汽车"更近。
    
    【TF-IDF 简介】
    TF-IDF = Term Frequency - Inverse Document Frequency（词频-逆文档频率）
    - TF：某个词在当前文档中出现次数
    - IDF：某个词在所有文档中的稀有程度
    - 优点：简单快速，无需训练
    - 缺点：不考虑词序和语义，对中文效果有限
    
    【为什么使用 TF-IDF？】
    - 本演示使用是为了简化依赖，不需要外部 API
    - 实际生产中应使用更强大的 embedding 模型：
      * OpenAI text-embedding-3-small/3-large
      * DashScope text-embedding-v1/v2
      * HuggingFace 的 sentence-transformers
    
    【TF-IDF 对中文的处理】
    - analyzer='char_wb': 使用字符级分词（Character N-grams）
      * 将文本按字符切割，而非按词语
      * 对中文友好，因为中文没有明显的词边界
      * 例如："机器学习" → ["机", "器", "学", "习", "机器", "器学", "学习", ...]
    
    - ngram_range=(1, 3): 使用 1-3 元语法
      * 1-gram: 单字符 ["我", "爱", "中", "国"]
      * 2-gram: 连续两个字符 ["我爱", "爱中", "中国"]
      * 3-gram: 连续三个字符 ["我爱中", "爱中国"]
      * 这样可以捕获不同长度的词组信息
    
    - max_features=5000: 最多保留 5000 个特征
      * 限制向量维度，防止维度灾难
      * 保留最重要的 5000 个词/字符组合
    """

    def __init__(self):
        """
        初始化 TF-IDF 向量化器
        
        配置参数：
        - analyzer='char_wb': 字符级分词，wb 表示保留词边界（word boundary）
          对于中文，这意味着按每个字符处理，同时保留词的概念
        - ngram_range=(1, 3): 1-3 个字符的组合
        - max_features=5000: 最多 5000 维特征
        """
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # 字符级分词，对中文友好
            ngram_range=(1, 3),  # 1-3 gram
            max_features=5000,  # 最多 5000 维
        )
        self.fitted = False  # 标记是否已完成拟合（学习词汇表）

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入向量
        
        【调用时机】
        在创建向量存储之前调用，用于将所有文档转换为向量。
        
        【返回值】
        一个二维列表，每行是一个文档的向量表示。
        例如：[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        
        Args:
            texts: 文档文本列表，如 ["文档1内容", "文档2内容", ...]
        
        Returns:
            List[List[float]]: 每个文档的 TF-IDF 向量列表
        """
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True  # 标记已拟合，可以用于查询
        return vectors.toarray().tolist()  # 转换为普通 Python 列表

    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入向量
        
        【调用时机】
        用户提问时调用，将问题转换为向量。
        
        【重要】这里使用 transform 而非 fit_transform：
        - fit_transform: 学习词汇表 + 转换（用于文档）
        - transform: 只转换，不学习新词汇（用于查询）
        这样可以保证查询和文档使用相同的向量空间。
        
        Args:
            text: 用户查询文本，如 "LangChain 是什么"
        
        Returns:
            List[float]: 查询文本的向量表示
        """
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray()[0].tolist()

    def __call__(self, text: str) -> List[float]:
        """
        支持直接调用（函数式接口）
        
        【为什么需要这个方法？】
        LangChain 的部分组件要求传入一个可调用对象，
        而不是显式调用 embed_query。这个方法提供了这种兼容性。
        
        使用方式：
        >>> embeddings = TfidfEmbeddings()
        >>> vector = embeddings("要转换的文本")  # 直接调用
        
        Args:
            text: 要转换的文本
        
        Returns:
            List[float]: 文本的向量表示
        """
        return self.embed_query(text)


def get_embeddings() -> TfidfEmbeddings:
    """
    获取 Embedding 模型实例的工厂函数
    
    【设计模式：工厂模式】
    使用工厂函数而不是直接实例化，好处是：
    1. 统一入口，便于后续替换实现（如换用 OpenAI Embedding）
    2. 可以添加缓存、错误处理等逻辑
    3. 符合依赖注入原则，便于测试
    
    【返回类型】
    返回 TfidfEmbeddings 实例。
    如果要换成其他 embedding，只需要修改这个函数的返回值类型。
    
    Returns:
        TfidfEmbeddings: 配置好的 Embedding 实例
    
    Example:
        >>> embeddings = get_embeddings()
        >>> vectors = embeddings.embed_documents(["文档1", "文档2"])
        >>> query_vec = embeddings.embed_query("用户问题")
    """
    return TfidfEmbeddings()


def build_conversational_rag_chain(
    documents: List[Document],
    llm: ChatOpenAI,
    embeddings: TfidfEmbeddings,
    session_manager: ChatSessionManager,
) -> RunnableWithMessageHistory:
    """
    构建完整的 Conversational RAG 链
    
    ==========================================
    RAG 完整流程图
    ==========================================
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                        用户输入流程                              │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. 获取用户问题和会话历史                                        │
    │     - 从 session_manager 获取当前会话的所有历史消息               │
    │     - 将历史和当前问题一起处理                                     │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  2. 历史感知问题改写（History-Aware Query Rewriting）              │
    │     - 如果有历史消息，用 LLM 将问题改写为独立查询                   │
    │     - 例如: "它有哪些组件？" → "LangChain 有哪些核心组件？"        │
    │     - 如果没有历史，直接使用原问题                                  │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  3. 向量检索（Vector Retrieval）                                  │
    │     - 将改写后的问题转换为向量                                      │
    │     - 在 FAISS 向量库中搜索最相似的 k 个文档                       │
    │     - 返回相关文档列表                                             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  4. 答案生成（Answer Generation）                                 │
    │     - 将检索到的文档内容作为上下文                                  │
    │     - 将用户问题和历史消息一起发送给 LLM                            │
    │     - LLM 生成最终答案                                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  5. 会话历史更新                                                  │
    │     - 将用户问题和 AI 回答都添加到会话历史中                        │
    │     - 供下一轮对话使用                                             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                                    最终答案
    
    
    ==========================================
    详细步骤说明
    ==========================================
    
    【步骤 1-2: 文档处理与索引构建】
    - 使用 RecursiveCharacterTextSplitter 分割文档
    - 使用 FAISS 构建向量索引
    - 这是离线准备工作，只需要执行一次
    
    【步骤 3: 检索器创建】
    - vectorstore.as_retriever() 创建基础检索器
    - 支持 search_kwargs={"k": top_k} 设置返回数量
    
    【步骤 4-5: 提示词模板定义】
    - contextualize_q_prompt: 用于改写带指代词的问题
    - qa_prompt: 用于基于上下文生成答案
    
    【步骤 6-7: 历史感知检索链】
    - 使用 RunnableLambda 组合问题改写和检索
    - 实现上下文感知的检索逻辑
    
    【步骤 8: 会话包装】
    - 使用 RunnableWithMessageHistory 自动管理历史
    - 无需手动调用 session_manager.add_*
    
    ==========================================
    组件介绍
    ==========================================
    
    【RecursiveCharacterTextSplitter】
    递归字符分割器，按指定大小递归分割文本。
    优点：尽可能保持语义完整的段落不被切分。
    
    【FAISS】
    Facebook AI Similarity Search（Facebook AI 相似度搜索）
    - 高效的向量相似度搜索库
    - 支持百万级向量索引
    - 支持多种索引类型（Flat、IVF、HNSW 等）
    
    【RunnableLambda】
    LangChain 的函数式接口，用于将普通 Python 函数转换为 Runnable。
    这样可以将任意逻辑集成到 LangChain 的链式调用中。
    
    【RunnableWithMessageHistory】
    Runnable 包装器，自动管理会话历史。
    会在每次调用时注入历史消息，并在调用后自动保存新消息。
    
    Args:
        documents: 原始文档列表
            格式：List[Document]，Document 包含 page_content 和 metadata
            示例：[Document(page_content="...", metadata={"source": "..."})]
        
        llm: 语言模型实例
            用于问题改写和答案生成
            要求：支持 ChatOpenAI 接口的模型均可
        
        embeddings: 嵌入模型实例
            用于将文本转换为向量
            必须实现 embed_query 方法
        
        session_manager: 会话历史管理器
            用于存储和检索会话历史
            必须实现 get_session_history 方法
    
    Returns:
        RunnableWithMessageHistory: 配置好的对话式 RAG 链
            - 支持 invoke() 同步调用
            - 支持 ainvoke() 异步调用
            - 自动管理会话历史
    
    Example:
        >>> # 初始化组件
        >>> llm = ChatOpenAI(model="gpt-3.5-turbo")
        >>> embeddings = get_embeddings()
        >>> session_manager = ChatSessionManager()
        >>> documents = [Document(page_content="...")]
        >>> 
        >>> # 构建链
        >>> chain = build_conversational_rag_chain(
        ...     documents, llm, embeddings, session_manager
        ... )
        >>> 
        >>> # 调用链
        >>> response = chain.invoke(
        ...     {"input": "LangChain 是什么？"},
        ...     config={"configurable": {"session_id": "user123"}}
        ... )
        >>> print(response["answer"])
    """
    print("=" * 50)
    print("步骤 1: 文档分割")
    print("=" * 50)

    # =========================================================================
    # 【文档分割】将长文档切分成小块
    # =========================================================================
    # 
    # 【为什么需要分割文档？】
    # 1. LLM 有上下文长度限制，无法处理无限长的文本
    # 2. 检索粒度：块越小，检索越精准；块越大，上下文越完整
    # 3. 向量表示：每个块生成一个向量，块太多会增加存储和检索开销
    #
    # 【RecursiveCharacterTextSplitter 工作原理】
    # 1. 首先尝试按段落分割（由 separator 决定）
    # 2. 如果分割后的块仍然太大，继续递归分割
    # 3. 分割优先级：段落 → 句子 → 单词 → 字符
    #
    # 【关键参数】
    # - chunk_size: 每块的目标大小（字符数）
    # - chunk_overlap: 相邻块的重叠大小
    # - length_function: 计算文本长度的函数，默认是 len()
    #
    # 【chunk_overlap 的作用】
    # 假设文本 "ABCDEF"，chunk_size=4, chunk_overlap=2
    # 可能分割为：["ABCD", "CDEF"]，C 被两个块共享
    # 这样可以避免重要信息被切分到不同块
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # 每块 500 字符
        chunk_overlap=chunk_overlap, # 相邻块重叠 50 字符
        length_function=len,         # 使用 Python 的 len() 计算长度
    )
    splits = text_splitter.split_documents(documents)
    print(f"分割成 {len(splits)} 个文本块")

    print("\n" + "=" * 50)
    print("步骤 2: 创建向量存储 (FAISS)")
    print("=" * 50)

    # =========================================================================
    # 【向量存储构建】
    # =========================================================================
    #
    # 【为什么需要向量存储？】
    # 为了实现"语义搜索"而非简单的关键词搜索
    # 用户的查询 "如何训练模型" 可以匹配到 "模型训练方法" 等语义相近的内容
    #
    # 【FAISS 简介】
    # Facebook AI Similarity Search
    # - 专门用于高效向量相似度搜索的库
    # - 支持两种索引类型：
    #   * Flat (精确搜索): 暴力遍历所有向量，结果精确但速度慢
    #   * IVF/HNSW (近似搜索): 先聚类再搜索，速度快但可能有误差
    # - 对于小规模数据（<10000向量），Flat 索引足够
    #
    # 【FAISS.from_texts 参数】
    # - texts: 文本列表
    # - embedding: 嵌入函数，用于将文本转换为向量
    # - metadatas: 元数据列表，与文本一一对应
    
    # 提取文档文本和元数据
    # page_content: 文档的实际内容
    # metadata: 附加信息，如来源、标题等（用于溯源）
    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]

    # 对文档进行嵌入（转换为向量）
    # 这一步会学习文档的词汇表，建立向量空间
    embeddings.embed_documents(texts)

    # 创建 FAISS 向量存储
    # from_texts 会自动：
    # 1. 对每个文本调用 embeddings.embed_query()
    # 2. 构建向量索引
    # 3. 存储文本和元数据
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,  # 传入 embedding 对象用于查询
        metadatas=metadatas,
    )
    # vectorstore.index 是 FAISS 的索引对象
    # ntotal 属性表示索引中的向量数量
    print(f"向量存储创建完成，共 {vectorstore.index.ntotal} 个向量")

    print("\n" + "=" * 50)
    print("步骤 3: 创建检索器")
    print("=" * 50)

    # =========================================================================
    # 【检索器创建】
    # =========================================================================
    #
    # 【Retriever vs VectorStore】
    # - VectorStore: 向量存储的完整封装，支持多种操作
    # - Retriever: 专门用于检索的接口，更轻量
    #
    # 【as_retriever() 方法】
    # 将 VectorStore 转换为 Retriever 接口
    # 统一了不同向量存储的使用方式
    #
    # 【search_kwargs】
    # 传递给底层搜索方法的参数
    # - k: 返回最相似的 k 个结果
    
    # 创建基础检索器
    # as_retriever() 返回一个实现了 Retriever 接口的对象
    # search_kwargs={"k": top_k} 表示返回 top_k 个最相似的文档
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
    print(f"检索器创建完成，每次返回 {top_k} 个文档")

    print("\n" + "=" * 50)
    print("步骤 4: 定义提示词模板")
    print("=" * 50)

    # =========================================================================
    # 【提示词模板定义】
    # =========================================================================
    #
    # 【提示词工程（Prompt Engineering）的重要性】
    # 好的提示词可以让 LLM 更好地理解任务需求
    # LangChain 提供了 PromptTemplate 来管理提示词
    #
    # 【MessagesPlaceholder 的作用】
    # 这是一个占位符，在实际调用时会被替换为具体的消息列表
    # 实现了提示词的动态组装
    
    # -------------------------------------------------------------------------
    # 【历史感知问题改写提示词】
    # -------------------------------------------------------------------------
    # 
    # 【作用】
    # 在 Conversational RAG 中，用户可能使用指代词或省略句
    # 这个提示词让 LLM 将问题改写为独立完整的查询
    #
    # 【示例】
    # 对话历史：
    #   用户: "LangChain 是什么？"
    #   AI: "LangChain 是一个用于构建 LLM 应用的框架..."
    # 当前问题："它有哪些核心组件？"
    #
    # 改写后的问题："LangChain 有哪些核心组件？"
    #
    # 【设计要点】
    # 1. 明确说明输入：聊天历史 + 最新问题
    # 2. 明确说明任务：创建独立完整的问题
    # 3. 明确说明输出要求：只输出问题，不要解释
    
    contextualize_q_system_prompt = """给定聊天历史和最新的用户问题，
该问题可能引用聊天历史中的上下文（如"它"、"那个"等指代词）。

请创建一个独立的、完整的问题，
使其可以在不需要聊天历史的情况下被理解。

如果问题已经是独立的，直接原样返回。
只输出改写后的问题，不要有其他解释。
"""

    # 构建历史感知改写的提示词模板
    # 包含：系统指令 + 历史消息占位符 + 当前问题占位符
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),  # 系统指令
        MessagesPlaceholder(variable_name="chat_history"),  # 聊天历史
        ("human", "{input}"),  # 用户当前问题
    ])

    # -------------------------------------------------------------------------
    # 【问答生成提示词】
    # -------------------------------------------------------------------------
    #
    # 【作用】
    # 让 LLM 基于检索到的上下文回答用户问题
    #
    # 【设计要点】
    # 1. 明确 LLM 的角色："用于问答任务的助手"
    # 2. 提供上下文：{context} 占位符，会被替换为检索结果
    # 3. 设置回答策略：
    #    - 基于上下文回答
    #    - 不知道就说不知道（避免幻觉）
    #    - 使用中文
    
    system_prompt = """你是一个用于问答任务的助手。

根据以下检索到的上下文片段来回答用户的问题。
如果你在上下文中找不到答案，请直接说"我在当前文档中没有找到相关信息"。
请用中文回答，语言要自然流畅。

上下文：
{context}
"""

    # 构建问答提示词模板
    # optional=True 表示如果历史为空，可以不提供 chat_history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # 系统指令
        MessagesPlaceholder(variable_name="chat_history", optional=True),  # 历史（可选）
        ("human", "{input}"),  # 用户问题
    ])

    print("提示词模板已定义")

    print("\n" + "=" * 50)
    print("步骤 5: 创建历史感知检索链")
    print("=" * 50)

    # =========================================================================
    # 【历史感知检索链】
    # =========================================================================
    #
    # 【核心思想】
    # 将"问题改写"和"向量检索"串联起来
    # 改写后的独立问题 → 向量检索 → 相关文档
    #
    # 【为什么需要改写？】
    # 向量检索基于语义相似度
    # "它有哪些组件？" 与 "LangChain 组件" 的向量相似度可能较低
    # 改写后变成 "LangChain 有哪些组件？" 就能正确检索
    #
    # 【RunnableLambda 的作用】
    # 将普通 Python 函数包装成 Runnable
    # 使得函数可以参与 LangChain 的链式调用
    
    def contextualize_question(inputs: dict) -> dict:
        """
        将问题改写为独立查询（历史感知检索的核心）
        
        【输入】
        - inputs["input"]: 用户当前问题
        - inputs["chat_history"]: 聊天历史列表
        
        【输出】
        - 包含独立问题的字典
        
        【处理逻辑】
        1. 检查是否有聊天历史
        2. 如果没有，直接返回原问题（无需改写）
        3. 如果有，调用 LLM 改写问题
        
        【为什么用 RunnableLambda 包装？】
        - 实现函数式接口，可以参与链式调用
        - 支持异步调用（可通过 ainvoke）
        - 集成 LangChain 的回调系统
        """
        # 从输入中获取聊天历史和用户问题
        chat_history = inputs.get("chat_history", [])  # 获取历史，默认空列表
        user_input = inputs["input"]  # 用户问题

        # 如果没有历史消息，直接返回原问题
        # 这是优化：不需要无谓地调用 LLM
        if not chat_history:
            return {"input": user_input}

        # 有历史消息，需要改写问题
        # 调用 LLM 进行问题改写
        # format() 方法会替换提示词中的占位符
        context_prompt = contextualize_q_prompt.format(
            chat_history=chat_history,  # 替换 {chat_history}
            input=user_input           # 替换 {input}
        )
        
        # 调用 LLM 获取改写后的问题
        response = llm.invoke(context_prompt)
        
        # 返回改写后的问题（去掉可能的空白字符）
        return {"input": response.content.strip()}

    # 使用管道操作符 | 组合两个 Runnable
    # 语义等价于：result = base_retriever.invoke(contextualize_question(inputs))
    #
    # 【管道操作符 | 的工作原理】
    # 输入 → contextualize_question() → 输出 → base_retriever.invoke() → 最终结果
    # 每个 | 表示将前一个 Runnable 的输出传递给下一个 Runnable 作为输入
    
    history_aware_retriever_chain = (
        RunnableLambda(contextualize_question)  # 第一步：改写问题
        | base_retriever                        # 第二步：向量检索
    )
    print("历史感知检索链创建完成")

    print("\n" + "=" * 50)
    print("步骤 6: 创建问答生成链")
    print("=" * 50)

    # =========================================================================
    # 【问答生成链】
    # =========================================================================
    #
    # 【作用】
    # 根据检索到的文档和用户问题，生成最终答案
    #
    # 【与历史感知检索链的区别】
    # - 历史感知检索链：处理输入，改写问题 + 检索文档
    # - 问答生成链：处理输出，格式化上下文 + 生成答案
    #
    # 【注意】
    # 这个函数在当前代码中定义但实际未直接使用（被 full_rag_chain 替代）
    # 这里保留是为了展示如何构建完整的问答链
    
    def generate_answer(inputs: dict) -> dict:
        """
        基于上下文生成答案
        
        【输入】
        - inputs["context"]: 检索到的文档列表
        - inputs["input"]: 用户问题
        - inputs["chat_history"]: 聊天历史（可选）
        
        【输出】
        - 包含答案和上下文的字典
        
        【处理流程】
        1. 提取检索到的文档内容
        2. 拼接成上下文字符串
        3. 调用 LLM 生成答案
        4. 返回结果
        """
        docs = inputs.get("context", [])  # 获取检索结果
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        # 构建上下文字符串
        # 每个文档的 page_content 用换行分隔
        # 如果没有文档，返回提示信息
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "没有找到相关文档。"

        # 调用 LLM 生成答案
        answer_prompt = qa_prompt.format(
            context=context,          # 替换 {context}
            chat_history=chat_history, # 替换 {chat_history}
            input=user_input          # 替换 {input}
        )
        response = llm.invoke(answer_prompt)

        # 返回答案和使用的文档（便于后续分析或溯源）
        return {
            "answer": response.content,
            "context": docs  # 返回原始文档，用于调试或展示来源
        }

    qa_chain = RunnableLambda(generate_answer)
    print("问答生成链创建完成")

    print("\n" + "=" * 50)
    print("步骤 7: 创建完整 RAG 链")
    print("=" * 50)

    # =========================================================================
    # 【完整 RAG 链】
    # =========================================================================
    #
    # 【整合所有组件】
    # 完整的 RAG 流程 = 检索 + 生成
    # 
    # 【为什么不直接用 history_aware_retriever_chain | qa_chain？】
    # 因为 qa_chain 需要 "context" 键，而检索器返回的是文档列表
    # 需要额外处理来将文档列表转换为 "context" 键
    #
    # 【当前实现的简化】
    # 为了简化，这里将检索和生成合并到一个函数中
    # 在实际生产中，可以使用 LCEL 的更优雅的实现方式
    
    def full_rag_chain(inputs: dict) -> dict:
        """
        完整的 RAG 处理流程
        
        【流程】
        1. 用户输入 → 历史感知检索 → 相关文档
        2. 相关文档 + 用户输入 → LLM → 最终答案
        
        【为什么不分离改写和检索？】
        这里为了简化，直接用 base_retriever 检索
        实际生产中应该用 history_aware_retriever_chain
        来实现完整的历史感知功能
        
        【inputs 包含】
        - input: 用户问题
        - chat_history: 聊天历史（由 RunnableWithMessageHistory 注入）
        """
        # 第一步：检索相关文档
        # 使用基础检索器（未使用历史感知改写，这里可以改进）
        docs = base_retriever.invoke(inputs["input"])
        
        # 构建上下文字符串
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "没有找到相关文档。"

        # 第二步：生成答案
        # 调用 LLM，基于上下文回答问题
        answer_prompt = qa_prompt.format(
            context=context,
            chat_history=[],  # 这里简化处理，历史由外层管理
            input=inputs["input"]
        )
        response = llm.invoke(answer_prompt)

        # 返回结果
        # answer: 最终答案
        # context: 使用的文档（便于溯源）
        return {
            "answer": response.content,
            "context": docs
        }

    rag_chain = RunnableLambda(full_rag_chain)
    print("完整 RAG 链创建完成")

    print("\n" + "=" * 50)
    print("步骤 8: 添加会话历史管理")
    print("=" * 50)

    # =========================================================================
    # 【会话历史管理】
    # =========================================================================
    #
    # 【RunnableWithMessageHistory 的作用】
    # 这是一个 Runnable 包装器，自动管理会话历史
    #
    # 【工作原理】
    # 1. 调用链之前：从 session_manager 获取历史，注入到输入中
    # 2. 调用链之后：将用户输入和 AI 输出都保存到 session_manager
    #
    # 【参数说明】
    # - chain: 要包装的链
    # - get_session_history: 获取会话历史的函数
    # - input_messages_key: 用户输入在输入字典中的键名
    # - history_messages_key: 历史消息在提示词中的占位符名
    # - output_messages_key: AI 输出在输出字典中的键名（用于保存到历史）
    #
    # 【为什么需要 RunnableWithMessageHistory？】
    # 1. 自动化：无需手动管理历史，降低出错概率
    # 2. 统一接口：会话和非会话场景使用相同的调用方式
    # 3. 多会话支持：通过 session_id 隔离不同用户/对话
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,                            # 要包装的 RAG 链
        session_manager.get_session_history,  # 获取会话历史的回调
        input_messages_key="input",           # 用户输入的键名
        history_messages_key="chat_history", # 历史消息占位符
        output_messages_key="answer",         # AI 输出的键名（用于保存）
    )
    print("会话历史管理添加完成")

    print("\n" + "=" * 50)
    print("Conversational RAG 链构建完成！")
    print("=" * 50)

    return conversational_rag_chain


def demo_conversational_rag():
    """
    完整 Conversational RAG 演示
    
    ==========================================
    演示流程说明
    ==========================================
    
    本函数演示一个完整的多轮对话场景：
    1. 用户询问 "LangChain 是什么？" → AI 回答
    2. 用户追问 "它有哪些核心组件？" → AI 理解"它"指 LangChain
    3. 用户询问 "什么是 RAG？"
    4. 用户追问 " Conversational RAG 有什么优势？" → AI 理解是在比较两种 RAG
    
    【关键点】
    - 每轮对话之间保持上下文
    - AI 能理解指代词（"它"）
    - AI 能理解省略句（"相比普通的 RAG..."）
    - 答案基于预定义的文档内容生成
    """
    print("\n" + "=" * 60)
    print("LangChain Conversational RAG 完整演示")
    print("=" * 60)

    # 初始化组件
    llm = model  # 使用全局模型
    embeddings = get_embeddings()
    session_manager = ChatSessionManager()

    # =========================================================================
    # 【预定义演示文档】
    # =========================================================================
    # 
    # 这里使用硬编码的文档作为演示
    # 实际应用中可以从文件、数据库、API 等多种来源加载文档
    
    # 文档1: LangChain 简介
    documents = [
        Document(
            page_content="""
            LangChain 是一个用于构建 LLM（大型语言模型）应用的开发框架。

            LangChain 提供了丰富的组件，包括：
            - 模型调用（Models）：支持多种 LLM 提供商
            - 提示词管理（Prompts）：模板化提示词
            - 索引（Indexes）：文档加载和分割
            - 链（Chains）：组合多个组件
            - 代理（Agents）：让 LLM 调用工具
            - 内存（Memory）：管理对话历史

            使用 LangChain 可以快速构建复杂的 LLM 应用，如 RAG 系统、聊天机器人等。
            """,
            metadata={"source": "langchain_intro.txt", "category": "framework"}
        ),
        
        # 文档2: Agent 概念
        Document(
            page_content="""
            Agent（代理）是 LangChain 中的核心概念。

            Agent 可以使用工具来执行复杂任务。常见类型包括：
            - ReAct 代理：结合推理(Reasoning)和行动(Action)的代理模式
            - OpenAI 代理：使用 OpenAI API 的代理
            - SQL 代理：用于数据库查询的代理
            - Conversational 代理：用于对话系统的代理

            Agent 的工作流程：
            1. LLM 分析用户输入
            2. 选择并调用合适的工具
            3. 获取工具返回的结果
            4. 根据结果决定下一步行动
            5. 最终返回答案给用户
            """,
            metadata={"source": "agent_concept.txt", "category": "agent"}
        ),
        
        # 文档3: RAG 基础
        Document(
            page_content="""
            RAG（检索增强生成，Retrieval-Augmented Generation）是一种结合检索和生成的架构。

            RAG 的核心思想：
            1. 从外部知识库检索相关信息
            2. 将检索结果作为上下文提供给 LLM
            3. LLM 基于最新和最相关的信息生成答案

            RAG 的优势：
            - 解决 LLM 知识过时的问题
            - 支持最新信息的问答
            - 减少幻觉（Hallucination）
            - 可追溯答案来源
            - 保护私有数据
            """,
            metadata={"source": "rag_basics.txt", "category": "rag"}
        ),
        
        # 文档4: Conversational RAG 进阶
        Document(
            page_content="""
            Conversational RAG 是 RAG 的进阶版本，专门优化用于多轮对话场景。

            核心挑战：
            - 用户可能使用代词（它、那个）指代之前的话题
            - 问题可能是省略句，需要结合上下文理解
            - 需要追踪和维护对话历史

            解决方案：
            - History-Aware Retriever：历史感知检索器
            - 会话状态管理：Session Management
            - 上下文重写：Context Rewriting

            History-Aware Retriever 的工作原理：
            1. 接收用户当前问题和对话历史
            2. 使用 LLM 将问题改写为独立查询
            3. 用独立查询检索相关文档
            4. 将原始问题和检索结果一起返回

            这使得系统能够理解"它指的是什么"这类上下文依赖的问题。
            """,
            metadata={"source": "conversational_rag.txt", "category": "rag"}
        ),
    ]

    print(f"使用 {len(documents)} 个预定义文档进行演示\n")

    # =========================================================================
    # 【构建 RAG 链】
    # =========================================================================
    # 这一步会：
    # 1. 分割文档
    # 2. 生成向量
    # 3. 构建 FAISS 索引
    # 4. 创建检索器和链
    # 5. 添加会话管理
    
    rag_chain = build_conversational_rag_chain(
        documents=documents,
        llm=llm,
        embeddings=embeddings,
        session_manager=session_manager,
    )

    # =====================================================================
    # 多轮对话演示
    # =====================================================================
    # 
    # 【关键机制】
    # - 所有轮次使用相同的 session_id
    # - session_manager 自动累积历史
    # - 每次调用返回的 response 包含：
    #   * answer: AI 的回答
    #   * context: 检索到的文档（用于调试）
    
    session_id = "demo_session"

    print("\n" + "=" * 60)
    print("开始对话演示")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 【第一轮对话】简单问答
    # -------------------------------------------------------------------------
    print("\n--- 第一轮对话 ---")
    print("用户: LangChain 是什么？")

    # 调用 RAG 链
    # config={"configurable": {"session_id": session_id}} 指定会话 ID
    response = rag_chain.invoke(
        {"input": "LangChain 是什么？"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"AI: {response['answer']}")

    # -------------------------------------------------------------------------
    # 【第二轮对话】使用指代词
    # -------------------------------------------------------------------------
    print("\n--- 第二轮对话 ---")
    print("用户: 它有哪些核心组件？")
    
    # 关键点：这里使用了"它"来指代第一轮中的 LangChain
    # 系统需要结合历史理解"它"指的是"LangChain"
    
    response = rag_chain.invoke(
        {"input": "它有哪些核心组件？"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"AI: {response['answer']}")

    # -------------------------------------------------------------------------
    # 【第三轮对话】询问另一个主题
    # -------------------------------------------------------------------------
    print("\n--- 第三轮对话 ---")
    print("用户: 什么是 RAG？")
    
    # 切换到新话题，但仍保持历史
    # AI 应该能区分这是在问 RAG 的概念
    
    response = rag_chain.invoke(
        {"input": "什么是 RAG？"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"AI: {response['answer']}")

    # -------------------------------------------------------------------------
    # 【第四轮对话】比较式提问
    # -------------------------------------------------------------------------
    print("\n--- 第四轮对话 ---")
    print("用户: 相比普通的 RAG，Conversational RAG 有什么优势？")
    
    # 这是一个比较式的问题
    # 需要理解"普通的 RAG"指的是什么
    # "Conversational RAG" 的优势需要结合之前的对话理解
    
    response = rag_chain.invoke(
        {"input": "相比普通的 RAG，Conversational RAG 有什么优势？"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"AI: {response['answer']}")

    # -------------------------------------------------------------------------
    # 【显示会话历史】
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("会话历史记录")
    print("=" * 60)
    
    # 遍历会话中的所有消息
    for i, message in enumerate(session_manager.get_session_history(session_id).messages):
        role = "AI" if isinstance(message, AIMessage) else "用户"
        # 截断长消息，便于展示
        content = message.content[:80] + "..." if len(message.content) > 80 else message.content
        print(f"\n{i+1}. {role}: {content}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    """
    程序入口点
    
    ==========================================
    运行说明
    ==========================================
    
    1. 确保已安装依赖：
       pip install faiss-cpu langchain-text-splitters sklearn
    
    2. 配置环境变量（.env 文件）：
       LLM_MODEL_ID=qwen3.5-flash
       LLM_API_KEY=your_api_key
       LLM_BASE_URL=your_api_url
       LLM_TIMEOUT=60
       RAG_TOP_K=2
       CHUNK_SIZE=500
       CHUNK_OVERLAP=50
    
    3. 运行程序：
       python session_retriever_all_in_one.py
    """
    
    # 显示当前配置
    print(f"模型配置: {model_name}")
    print(f"API 地址: {base_url}")
    print("=" * 50)

    # =========================================================================
    # 演示选择
    # =========================================================================
    # 
    # 以下演示函数按复杂度递增排序：
    # 1. demo_chat_session_manager: 最简单，只演示会话管理
    # 2. demo_runnable_with_message_history: 演示 LangChain 的会话管理封装
    # 3. demo_custom_retriever: 演示如何创建自定义检索器
    # 4. demo_conversational_rag: 完整 RAG 演示（默认运行）
    
    # # 会话管理示例 - 最基础的会话管理方式
    # demo_chat_session_manager()
    #
    # # RunnableWithMessageHistory 示例 - LangChain 推荐的会话管理方式
    # demo_runnable_with_message_history()
    #
    # # 自定义 Retriever 示例 - 展示如何扩展 LangChain
    # demo_custom_retriever()

    # 完整 Conversational RAG 演示 - 综合所有组件
    demo_conversational_rag()
