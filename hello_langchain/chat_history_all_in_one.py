"""
LangChain 消息管理与聊天历史存储完整示例

包含以下功能:
1. 内存消息历史存储
2. 自定义会话键 (user_id + conversation_id)
3. Redis 持久化存储
4. 消息裁剪 (限制保留消息数量)
5. 消息摘要 (使用 LLM 压缩历史)

运行方式:
    python chat_history_all_in_one.py

    # 或指定模式
    python chat_history_all_in_one.py --mode memory
    python chat_history_all_in_one.py --mode configurable
    python chat_history_all_in_one.py --mode redis
    python chat_history_all_in_one.py --mode trim
    python chat_history_all_in_one.py --mode summarize
"""

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    import os
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    pass

import os
import argparse
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langchain_core.runnables.history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# ==================== 公共组件 ====================

# 创建聊天模型实例
# 使用阿里云通义千问模型 (兼容 OpenAI API)
# 从环境变量读取配置: LLM_MODEL_ID, LLM_API_KEY, LLM_BASE_URL, LLM_TIMEOUT
chat = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "qwen3.5-flash"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    timeout=int(os.getenv("LLM_TIMEOUT", "60"))
)

# 创建提示模板
# ChatPromptTemplate: 用于构建聊天提示的模板类
# MessagesPlaceholder: 占位符，用于动态插入消息历史
# 参数说明:
#   - system: 系统消息，定义助手的行为和性格
#   - MessagesPlaceholder(variable_name="history"): 会被替换为历史消息
#   - human: 用户输入，{input} 是占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手，尽力回答所有问题。"),
    # history 变量将包含之前的对话历史 (可选)
    MessagesPlaceholder(variable_name="history", optional=True),
    # chat_history 变量也可以用于存储历史消息 (可选)
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    # 用户的当前输入
    ("human", "{input}"),
])

# 创建链 (Chain)
# 使用 | 运算符将多个组件串联起来
# 数据流向: input -> prompt.format() -> chat.invoke() -> output
chain = prompt | chat


# ==================== 1. 内存消息历史 ====================
# 演示最简单的消息历史存储方式：使用内存字典
# 优点：简单快速 缺点：程序重启后消息丢失

class InMemoryHistoryStore:
    """
    内存消息历史存储类
    
    使用 Python 字典在内存中存储各会话的消息历史。
    适用于：单进程、不需要持久化的场景。
    
    原理:
    - session_id: 会话的唯一标识符（如用户ID、对话ID）
    - ChatMessageHistory: LangChain 提供的消息历史类，
      自动管理消息列表（用户消息 + AI 消息）
    """
    
    def __init__(self):
        # 存储结构: {session_id: ChatMessageHistory实例}
        self.store = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        获取指定会话的历史消息
        
        Args:
            session_id: 会话的唯一标识符
            
        Returns:
            BaseChatMessageHistory: 该会话的消息历史对象
        """
        # 如果会话不存在，创建新的 ChatMessageHistory
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        # 返回该会话的历史记录
        return self.store[session_id]


def demo_memory_history():
    """演示内存消息历史的基本用法"""
    print("=" * 60)
    print("1. 内存消息历史")
    print("=" * 60)
    
    # 创建内存存储实例
    store = InMemoryHistoryStore()
    
    # RunnableWithMessageHistory: LangChain 提供的包装器
    # 作用：自动为链添加消息历史管理功能
    # 参数说明:
    #   - chain: 基础的对话链（prompt | chat）
    #   - store.get_session_history: 获取会话历史的函数
    #   - input_messages_key: 用户输入的键名（对应 prompt 中的 {input}）
    #   - history_messages_key: 历史消息的键名（对应 MessagesPlaceholder）
    with_message_history = RunnableWithMessageHistory(
        chain,
        store.get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # 第一次对话 - 初始化会话
    # config 参数用于传递会话标识
    # session_id="abc123" 表示这是一个独立的会话
    print("\n--- 第一次对话 (session_id: abc123) ---")
    response = with_message_history.invoke(
        {"input": "余弦是什么意思？"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(f"用户: 余弦是什么意思？")
    print(f"AI: {response.content}")
    
    # 第二次对话 - 使用相同的 session_id
    # RunnableWithMessageHistory 会自动:
    # 1. 调用 get_session_history("abc123") 获取历史
    # 2. 将历史消息填充到 MessagesPlaceholder
    # 3. 将用户新输入与历史一起发送给模型
    # 4. 将 AI 回复保存到历史中
    print("\n--- 第二次对话 (session_id: abc123) ---")
    response = with_message_history.invoke(
        {"input": "什么？"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(f"用户: 什么？")
    print(f"AI: {response.content}")
    
    # 新会话 - 使用不同的 session_id
    # session_id="def456" 是一个全新的会话
    # 历史存储是按 session_id 隔离的，所以不会记得 abc123 的对话
    print("\n--- 第三次对话 (session_id: def456 - 新会话) ---")
    response = with_message_history.invoke(
        {"input": "什么？"},
        config={"configurable": {"session_id": "def456"}},
    )
    print(f"用户: 什么？")
    print(f"AI: {response.content}")


# ==================== 2. 自定义会话键 ====================
# 场景：某些应用需要更复杂的会话标识方式
# 例如：一个用户可以有多个独立的对话（conversation_id）
# 通过自定义会话键，可以实现 user_id + conversation_id 的复合键

class ConfigurableHistoryStore:
    """
    可配置会话键的历史存储
    
    与简单 session_id 不同，这里使用 (user_id, conversation_id) 复合键。
    适用于：一个用户有多个独立对话场景的应用。
    
    复合键结构:
    - user_id: 用户ID，区分不同用户
    - conversation_id: 对话ID，区分同一用户的不同对话
    - 组合方式：将两个值打包成元组 (user_id, conversation_id)
    """
    
    def __init__(self):
        # 存储结构: {(user_id, conversation_id): ChatMessageHistory实例}
        self.store = {}
    
    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        """
        使用复合键获取会话历史
        
        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            
        Returns:
            对应的消息历史对象
        """
        # 将两个参数组合成元组作为键
        key = (user_id, conversation_id)
        if key not in self.store:
            self.store[key] = ChatMessageHistory()
        return self.store[key]


def demo_configurable_keys():
    """演示如何使用自定义的复合会话键"""
    print("\n" + "=" * 60)
    print("2. 自定义会话键 (user_id + conversation_id)")
    print("=" * 60)
    
    store = ConfigurableHistoryStore()
    
    # RunnableWithMessageHistory 的 history_factory_config 参数
    # 用于定义自定义会话键的结构（ConfigurableFieldSpec 列表）
    # 这样 LangChain 知道如何解析 config 中的参数
    
    # ConfigurableFieldSpec 参数说明:
    #   - id: 参数名（必须与 get_session_history 的参数名一致）
    #   - annotation: 类型注解（str 表示字符串）
    #   - name: 人类可读的名称（用于调试/文档）
    #   - description: 参数描述
    #   - default: 默认值
    #   - is_shared: 是否在多个调用间共享

    with_message_history = RunnableWithMessageHistory(
        chain,
        store.get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        # 定义两个自定义字段：user_id 和 conversation_id
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="用户的唯一标识符。",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="会话的唯一标识符。",
                default="",
                is_shared=True,
            ),
        ],
    )
    
    # 第一次对话 - 用户123开始一个新对话(对话ID=1)
    # 使用 config 中的 configurable 传递多个参数
    print("\n--- 第一次对话 (user_id: 123, conversation_id: 1) ---")
    response = with_message_history.invoke(
        {"input": "你好，我叫小明"},
        config={"configurable": {"user_id": "123", "conversation_id": "1"}},
    )
    print(f"AI: {response.content}")
    
    # 同一会话继续 - 相同的 user_id + conversation_id
    # 历史消息会被保留，模型可以看到之前的对话
    print("\n--- 第二次对话 (user_id: 123, conversation_id: 1) ---")
    response = with_message_history.invoke(
        {"input": "你还记得我叫什么吗？"},
        config={"configurable": {"user_id": "123", "conversation_id": "1"}},
    )
    print(f"AI: {response.content}")
    
    # 同一用户，新会话 - user_id 相同但 conversation_id 不同
    # 这是一个全新的对话，不记得之前 conversation_id=1 的内容
    print("\n--- 第三次对话 (user_id: 123, conversation_id: 2 - 新会话) ---")
    response = with_message_history.invoke(
        {"input": "你好"},
        config={"configurable": {"user_id": "123", "conversation_id": "2"}},
    )
    print(f"AI: {response.content}")


# ==================== 3. Redis 持久化 ====================
# 内存存储的缺点：程序重启后数据丢失
# Redis 持久化：消息历史存储在 Redis 数据库中，程序重启后仍可恢复
# 适用场景：生产环境、多进程部署、需要长期保存对话历史

def check_redis():
    """
    检查 Redis 连接是否可用
    
    Returns:
        tuple: (是否连接成功, Redis连接URL)
    """
    try:
        import redis
        # 从环境变量读取 Redis URL，默认使用本地 Redis
        REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:16379/0")
        # 创建 Redis 连接
        r = redis.from_url(REDIS_URL)
        # 发送 PING 命令检查连接
        r.ping()
        return True, REDIS_URL
    except Exception as e:
        print(f"Redis 连接失败: {e}")
        print("请确保 Redis 正在运行:")
        print("  docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest")
        return False, os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def demo_redis_history():
    """演示使用 Redis 存储消息历史"""
    print("\n" + "=" * 60)
    print("3. Redis 持久化消息历史")
    print("=" * 60)
    
    # 先检查 Redis 是否可用
    connected, REDIS_URL = check_redis()
    if not connected:
        return
    
    print(f"Redis URL: {REDIS_URL}")
    
    # 定义获取 Redis 历史记录的函数
    # RedisChatMessageHistory: LangChain 提供的 Redis 消息历史类
    # 会自动将消息序列化后存储到 Redis
    def get_redis_history(session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(session_id, url=REDIS_URL)
    
    # 创建带 Redis 持久化的对话链
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_redis_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # 使用固定的 session_id，这样可以在多次运行间保持对话历史
    session_id = "demo_redis_session"
    
    # 第一次对话 - 消息会被保存到 Redis
    print(f"\n--- 第一次对话 (session_id: {session_id}) ---")
    response = with_message_history.invoke(
        {"input": "余弦是什么意思？"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"用户: 余弦是什么意思？")
    print(f"AI: {response.content}")
    
    # 第二次对话 - 从 Redis 读取之前的历史
    # 即使程序重启，只要 session_id 相同，就能恢复之前的对话
    print(f"\n--- 第二次对话 (session_id: {session_id}) ---")
    response = with_message_history.invoke(
        {"input": "再说一遍？"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"用户: 再说一遍？")
    print(f"AI: {response.content}")
    
    # 手动查看 Redis 中存储的消息
    print("\n--- 查看 Redis 中的历史 ---")
    history = get_redis_history(session_id)
    for msg in history.messages:
        # 打印消息类型和内容（前50个字符）
        print(f"  - {type(msg).__name__}: {msg.content[:50]}...")


# ==================== 4. 消息裁剪 ====================
# 场景：长对话会产生大量历史消息，导致:
# 1. API 调用成本增加（按 token 收费）
# 2. 模型可能忽略早期重要信息
# 3. 超过模型的上下文窗口限制
# 
# 解决方案：消息裁剪 - 只保留最近 N 条消息
# 本示例演示如何实现自动裁剪，只保留最近 2 条消息

def demo_trim_messages():
    """演示消息裁剪功能"""
    print("\n" + "=" * 60)
    print("4. 消息裁剪 (只保留最近 2 条)")
    print("=" * 60)
    
    # 创建带预加载消息的历史（模拟已有6条消息的对话）
    # ChatMessageHistory 的消息结构是交替的：用户消息 -> AI消息 -> 用户消息 -> AI消息...
    temp_chat_history = ChatMessageHistory()
    temp_chat_history.add_user_message("我叫Jack，你好")
    temp_chat_history.add_ai_message("你好")
    temp_chat_history.add_user_message("我今天心情很开心")
    temp_chat_history.add_ai_message("你今天心情怎么样？")
    temp_chat_history.add_user_message("我下午在打篮球")
    temp_chat_history.add_ai_message("你下午在做什么？")
    
    # 提示模板 - 包含聊天历史
    # 使用 chat_history 作为历史消息的变量名
    trim_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的助手。提供的聊天历史包括与您交谈的用户的事实。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 构建裁剪链
    trim_chain = trim_prompt | chat
    
    # 创建消息存储函数
    def get_temp_history(session_id: str) -> ChatMessageHistory:
        return temp_chat_history
    
    # 创建带消息历史的链
    # 参数说明:
    #   - runnable: 基础的可运行对象（链）
    #   - get_session_history: 获取会话历史的函数
    #   - input_messages_key: 用户输入的键名
    #   - history_messages_key: 历史消息的键名
    chain_with_history = RunnableWithMessageHistory(
        runnable=trim_chain,
        get_session_history=get_temp_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    # 裁剪函数 - 定义消息裁剪的逻辑
    # RunnablePassthrough.assign 会将这个函数的返回值添加到输入中
    def trim_messages_fn(chain_input):
        """
        裁剪消息的函数
        
        逻辑：
        1. 检查当前消息数量
        2. 如果超过限制，保留最近 N 条
        3. 返回 True 表示执行了裁剪
        """
        stored_messages = temp_chat_history.messages
        # 如果消息数 <= 2，不需要裁剪
        if len(stored_messages) <= 2:
            return False
        print(f"\n裁剪前: {len(stored_messages)} 条消息")
        # 清空所有消息
        temp_chat_history.clear()
        # 只保留最近 2 条消息（用户消息 + AI 回复）
        for message in stored_messages[-2:]:
            temp_chat_history.add_message(message)
        print(f"裁剪后: {len(temp_chat_history.messages)} 条消息")
        return True
    
    # 构建带裁剪功能的链
    # 流程: input -> trim_messages_fn(裁剪历史) -> chain_with_history(生成回复)
    # RunnablePassthrough.assign 的作用:
    #   - 先执行 trim_messages_fn 修改历史消息
    #   - 然后将结果传递给 chain_with_history
    chain_with_trimming = (
        RunnablePassthrough.assign(messages_trimmed=trim_messages_fn)
        | chain_with_history
    )
    
    # 打印初始消息
    print("\n初始消息:")
    for msg in temp_chat_history.messages:
        print(f"  - {msg.content[:30]}...")
    
    # 对话1 - 不触发裁剪（当前 6 条 > 2，但第一次只保留 2 条）
    # 注意：这里只调用 chain_with_history，不走裁剪逻辑
    print("\n--- 对话1 (不触发裁剪，消息数 <= 2) ---")
    response = chain_with_history.invoke(
        {"input": "我今天心情如何？"},
        config={"configurable": {"session_id": "unused"}},
    )
    print(f"AI: {response.content}")
    
    # 对话2 - 触发裁剪（使用 chain_with_trimming）
    # 裁剪函数会执行，只保留最近 2 条消息
    print("\n--- 对话2 (触发裁剪，只保留最近 2 条) ---")
    response = chain_with_trimming.invoke(
        {"input": "我下午在做什么？"},
        config={"configurable": {"session_id": "unused"}},
    )
    print(f"AI: {response.content}")
    
    # 对话3 - 继续对话（此时历史消息已被裁剪）
    print("\n--- 对话3 (继续对话) ---")
    response = chain_with_trimming.invoke(
        {"input": "我叫什么名字？"},
        config={"configurable": {"session_id": "unused"}},
    )
    print(f"AI: {response.content}")
    
    # 验证最终消息数量
    print("\n最终消息 (只剩 2 条):")
    for msg in temp_chat_history.messages:
        print(f"  - {msg.content[:30]}...")


# ==================== 5. 消息摘要 ====================
# 场景：与消息裁剪不同，摘要保留了更多上下文信息
# 消息裁剪：直接丢弃旧消息，可能丢失重要信息
# 消息摘要：使用 LLM 将旧消息压缩成摘要，保留关键信息
# 
# 优点：
# 1. 节省 token（比裁剪更高效）
# 2. 保留关键上下文（人名、事件、偏好等）
# 3. 模型仍可了解对话全貌
#
# 缺点：
# 1. 需要额外的 LLM 调用（摘要生成）
# 2. 摘要可能丢失细节

def demo_summarize_messages():
    """演示使用 LLM 生成消息摘要"""
    print("\n" + "=" * 60)
    print("5. 消息摘要 (使用 LLM 压缩历史)")
    print("=" * 60)
    
    # 创建带预加载消息的历史（模拟已有6条消息的对话）
    temp_chat_history = ChatMessageHistory()
    temp_chat_history.add_user_message("我叫Alice，你好")
    temp_chat_history.add_ai_message("你好 Alice")
    temp_chat_history.add_user_message("我今天心情很开心")
    temp_chat_history.add_ai_message("你今天心情怎么样？")
    temp_chat_history.add_user_message("我下午在打篮球")
    temp_chat_history.add_ai_message("你下午在做什么？")
    
    # 提示模板 - 用于生成最终回复
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 构建摘要链
    summary_chain = summary_prompt | chat
    
    # 创建消息存储函数
    def get_temp_history(session_id: str) -> ChatMessageHistory:
        return temp_chat_history
    
    # 创建带消息历史的链
    chain_with_history = RunnableWithMessageHistory(
        runnable=summary_chain,
        get_session_history=get_temp_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    # 摘要生成函数 - 核心逻辑
    def summarize_fn(chain_input):
        """
        使用 LLM 生成消息摘要
        
        工作流程：
        1. 获取当前所有消息
        2. 调用 LLM 将消息压缩成摘要
        3. 清空历史，保存摘要
        """
        stored_messages = temp_chat_history.messages
        # 如果没有消息，不需要摘要
        if len(stored_messages) == 0:
            return False
        
        print(f"\n摘要前: {len(stored_messages)} 条消息")
        
        # 创建摘要提示 - 指示 LLM 压缩消息
        # 关键点：让 LLM 保留具体事实（人名、事件、偏好等）
        summarization_prompt = ChatPromptTemplate.from_messages([
            ("user", "将上面的聊天消息压缩成一条简要消息，尽可能包含具体事实。"),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        
        # 构建摘要链并调用
        summarization_chain = summarization_prompt | chat
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
        
        print(f"摘要内容: {summary_message.content}")
        
        # 清空历史消息，保存摘要
        # 这样下次对话时只发送摘要，节省 token
        temp_chat_history.clear()
        temp_chat_history.add_message(summary_message)
        
        print(f"摘要后: {len(temp_chat_history.messages)} 条消息")
        return True
    
    # 构建带摘要功能的链
    # 流程: input -> summarize_fn(生成摘要) -> chain_with_history(生成回复)
    chain_with_summarization = (
        RunnablePassthrough.assign(messages_summarized=summarize_fn)
        | chain_with_history
    )
    
    # 打印初始消息
    print("\n初始消息:")
    for msg in temp_chat_history.messages:
        print(f"  - {msg.content[:30]}...")
    
    # 对话1 - 触发摘要
    # 第一次调用会调用 LLM 生成摘要，然后生成回复
    print("\n--- 对话1 (触发摘要) ---")
    response = chain_with_summarization.invoke(
        {"input": "我叫什么名字？"},
        config={"configurable": {"session_id": "unused"}},
    )
    print(f"AI: {response.content}")
    
    # 手动添加新消息 - 模拟用户继续对话
    print("\n--- 添加新消息后 ---")
    temp_chat_history.add_user_message("我喜欢画画")
    temp_chat_history.add_ai_message("画画很有趣")
    for msg in temp_chat_history.messages:
        print(f"  - {msg.content[:50]}...")
    
    # 对话2 - 再次触发摘要
    # 这次会包含之前的摘要 + 新消息，生成新的摘要
    print("\n--- 对话2 (再次触发摘要，包含之前的摘要) ---")
    response = chain_with_summarization.invoke(
        {"input": "我的爱好是什么？"},
        config={"configurable": {"session_id": "unused"}},
    )
    print(f"AI: {response.content}")
    
    # 查看最终消息（已被摘要）
    print("\n最终消息 (已被摘要):")
    for msg in temp_chat_history.messages:
        print(f"  - {msg.content}")


# ==================== 主函数 ====================
# 程序入口点，支持通过命令行参数选择不同的演示模式

def main():
    """
    主函数 - 解析命令行参数并运行相应的演示
    
    支持的模式:
    - memory: 内存消息历史
    - configurable: 自定义会话键
    - redis: Redis 持久化
    - trim: 消息裁剪
    - summarize: 消息摘要
    - all: 运行所有演示（默认）
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="LangChain 消息管理与聊天历史存储示例")
    
    # 添加 --mode 参数
    parser.add_argument(
        "--mode",
        # 可选的运行模式
        choices=["memory", "configurable", "redis", "trim", "summarize", "all"],
        default="all",  # 默认运行所有模式
        help="选择运行模式"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 打印标题
    print("=" * 60)
    print("LangChain 消息管理与聊天历史存储")
    print("=" * 60)
    
    # 根据选择的模式运行相应的演示函数
    if args.mode == "memory":
        demo_memory_history()
    elif args.mode == "configurable":
        demo_configurable_keys()
    elif args.mode == "redis":
        demo_redis_history()
    elif args.mode == "trim":
        demo_trim_messages()
    elif args.mode == "summarize":
        demo_summarize_messages()
    elif args.mode == "all":
        # 运行所有演示
        demo_memory_history()
        demo_configurable_keys()
        demo_redis_history()
        demo_trim_messages()
        demo_summarize_messages()


# 程序入口点
if __name__ == "__main__":
    main()