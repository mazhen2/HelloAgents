"""
LangChain 自定义 Callback 组件示例
包含：回调概念、事件类型、回调处理程序、运行时传递回调、自定义Handler

本文件演示 LangChain 中 Callback 机制的核心用法：
1. BaseCallbackHandler: 自定义回调处理程序的基础类
2. 运行时传递回调: 通过 config 参数传递 callbacks
3. 构造函数回调: 在初始化时传入 callbacks 参数
4. 流式输出: 使用 on_llm_new_token 捕获流式 token
5. 完整事件追踪: 追踪 Chain/Agent/Tool 的完整执行过程

Callback 机制允许我们在 LLM、Chain、Agent、Tool 的各个执行阶段
插入自定义逻辑，实现日志记录、性能监控、进度展示等功能。
"""

import os
from typing import Dict, Any, List

# 从 .env 文件加载环境变量（如 API Keys）
from dotenv import load_dotenv

# LangChain 核心回调组件
# BaseCallbackHandler: 自定义回调处理程序的基础类
# 所有自定义 Handler 都应继承此类并重写相应的事件方法
from langchain_core.callbacks.base import BaseCallbackHandler

# 消息和输出相关的类型定义
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate

# OpenAI 兼容的 LLM 接口
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
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

# 初始化 LLM 模型实例
# 这是 LangChain 中的 Runnable 对象，可以与其他组件组合成 Chain
model = ChatOpenAI(
    model=model_name,  # 模型标识符
    api_key=api_key,  # API 认证
    base_url=base_url,  # 服务地址
    timeout=timeout,  # 请求超时
)

# =============================================================================
# 回调事件 (Callback Events) 概述
# =============================================================================

"""
| Event                    | Event Trigger                        | Associated Method      |
|--------------------------|--------------------------------------|-----------------------|
| Chat model start        | When a chat model starts             | on_chat_model_start   |
| LLM start               | When a LLM starts                   | on_llm_start          |
| LLM new token           | When LLM emits a new token           | on_llm_new_token      |
| LLM ends                | When a LLM ends                      | on_llm_end            |
| LLM errors              | When a LLM errors                    | on_llm_error          |
| Chain start             | When a chain starts running          | on_chain_start        |
| Chain end               | When a chain ends                    | on_chain_end          |
| Chain error             | When a chain errors                  | on_chain_error        |
| Tool start              | When a tool starts running           | on_tool_start         |
| Tool end                | When a tool ends                     | on_tool_end           |
| Tool error              | When a tool errors                   | on_tool_error         |
| Agent action            | When an agent takes an action         | on_agent_action       |
| Agent finish            | When an agent ends                   | on_agent_finish       |
| Retriever start         | When a retriever starts               | on_retriever_start    |
| Retriever end           | When a retriever ends                 | on_retriever_end      |
| Retriever error         | When a retriever errors               | on_retriever_error    |
| Text                    | When arbitrary text is run           | on_text               |
| Retry                   | When a retry event is run            | on_retry              |
"""


# =============================================================================
# 示例1：基础 CallbackHandler 结构
# =============================================================================

class BaseCallbackHandler(BaseCallbackHandler):
    """
    基础回调处理程序，可处理 LangChain 的各类回调事件

    此示例展示 BaseCallbackHandler 的完整结构，
    所有回调方法默认为空实现 (pass)，你可以根据需要重写特定方法。
    只需实现你关心的事件方法，无需实现所有方法。
    """

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """
        当 LLM 开始运行时调用

        Args:
            serialized: LLM 模型的序列化信息（如模型名称、配置等）
            prompts: 输入的提示词列表
            **kwargs: 其他可选参数
        """
        pass

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """
        当聊天模型开始运行时调用

        Args:
            serialized: 模型序列化信息
            messages: 输入的消息列表（嵌套列表，每组代表一个对话历史）
            **kwargs: 其他可选参数
        """
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        当 LLM 生成新 token 时调用（流式输出）

        Args:
            token: 新生成的 token
            **kwargs: 其他可选参数
        """
        pass

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        当 LLM 结束运行时调用

        Args:
            response: LLM 的完整响应对象
            **kwargs: 其他可选参数
        """
        pass

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """
        当 LLM 发生错误时调用

        Args:
            error: 发生的异常对象
            **kwargs: 其他可选参数
        """
        pass

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """
        当链开始运行时调用

        Args:
            serialized: 链的序列化信息
            inputs: 链的输入参数
            **kwargs: 其他可选参数
        """
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """
        当链结束运行时调用

        Args:
            outputs: 链的输出结果
            **kwargs: 其他可选参数
        """
        pass

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """
        当链发生错误时调用

        Args:
            error: 发生的异常对象
            **kwargs: 其他可选参数
        """
        pass

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """
        当工具开始运行时调用

        Args:
            serialized: 工具的序列化信息
            input_str: 工具的输入参数
            **kwargs: 其他可选参数
        """
        pass

    def on_tool_end(self, output: str, **kwargs) -> None:
        """
        当工具结束运行时调用

        Args:
            output: 工具的输出结果
            **kwargs: 其他可选参数
        """
        pass

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """
        当工具发生错误时调用

        Args:
            error: 发生的异常对象
            **kwargs: 其他可选参数
        """
        pass


# =============================================================================
# 示例2：日志记录 Handler（运行时传递回调）
# =============================================================================

class LoggingHandler(BaseCallbackHandler):
    """
    日志记录回调处理程序

    此 Handler 演示如何记录 Chain 和 Chat Model 的执行过程。
    当你需要在调试时了解 Chain 的执行流程时，可以使用此类。
    """

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> None:
        """
        当聊天模型开始时调用

        打印模型启动信息，帮助追踪模型何时被调用。
        """
        print(">>> Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        当 LLM 结束时调用

        打印完整的响应对象，便于调试和日志记录。
        """
        print(f">>> Chat model ended, response: {response}")

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """
        当链开始时调用

        打印链的输入参数，帮助追踪数据流向。
        """
        print(f">>> Chain started, inputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """
        当链结束时调用

        打印链的输出结果，帮助确认处理结果。
        """
        print(f">>> Chain ended, outputs: {outputs}")


def demo_runtime_callback():
    """
    运行时传递回调示例

    演示如何通过 config 参数在运行时传递回调处理程序。
    这种方式的优点是：
    1. 不影响模型本身的配置
    2. 可以在不同的调用中使用不同的 Handler
    3. 适合临时添加调试/日志功能
    """
    print("\n" + "=" * 50)
    print("1. 运行时传递回调函数")
    print("=" * 50)

    # 创建回调处理程序实例
    # 可以同时传递多个 Handler
    callbacks = [LoggingHandler()]

    # 创建提示模板，用于构造用户输入
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

    # 构建链：提示模板 -> 模型
    # 使用 | 操作符连接组件，这是 LangChain 的标准写法
    chain = prompt | model

    # 通过 config 传递回调（仅对该次请求生效）
    # config 是一个字典，可以包含 callbacks、metadata 等配置
    response = chain.invoke({"number": "2"}, config={"callbacks": callbacks})

    print(f"\n最终响应: {response.content}")


# =============================================================================
# 示例3：构造函数回调
# =============================================================================

def demo_constructor_callback():
    """
    构造函数回调示例

    演示如何通过构造函数传递回调处理程序。
    这种方式的优点是：
    1. 所有调用都会自动使用该 Handler
    2. 适合需要持续监控的场景（如 APM 集成）
    3. 代码更简洁，不需要每次调用都传递 config
    """
    print("\n" + "=" * 50)
    print("2. 构造函数回调")
    print("=" * 50)

    # 创建回调处理程序
    callbacks = [LoggingHandler()]

    # 在构造函数中传递回调（对该对象的所有调用生效）
    # 这样所有使用该 llm 的 Chain 调用都会触发回调
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        callbacks=callbacks
    )

    # 创建提示模板
    prompt = ChatPromptTemplate.from_template("What is 5 + {number}?")

    # 构建链
    chain = prompt | llm

    # 不需要再传递 callbacks，会自动使用构造函数中的回调
    response = chain.invoke({"number": "10"})
    print(f"\n最终响应: {response.content}")


# =============================================================================
# 示例4：自定义流式输出 Handler
# =============================================================================

class MyCustomHandler(BaseCallbackHandler):
    """
    自定义回调处理程序 - 用于流式输出

    此 Handler 演示如何捕获 LLM 的流式输出。
    通过 on_llm_new_token 方法，可以实时获取模型生成的每个 token，
    实现打字机效果或实时进度展示。
    """

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        当 LLM 生成新 token 时调用

        Args:
            token: 新生成的 token 片段
        """
        print(f"[Token] {token}", end="", flush=True)


def demo_streaming_callback():
    """
    自定义流式输出回调示例

    演示如何启用流式输出并实时显示 token。
    适用于需要即时反馈的用户界面。
    """
    print("\n" + "=" * 50)
    print("3. 自定义流式输出 Callback")
    print("=" * 50)

    # 创建提示模板，使用消息格式
    prompt = ChatPromptTemplate.from_messages([
        ("human", "给我讲个关于{animal}的笑话，限制20个字")
    ])

    # 启用流式处理的关键参数：
    # 1. streaming=True: 启用流式输出
    # 2. callbacks: 传入自定义 Handler 来处理每个 token
    streaming_model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        streaming=True,  # 启用流式输出
        callbacks=[MyCustomHandler()]  # 传入 token 处理 Handler
    )

    # 构建链
    chain = prompt | streaming_model

    # 执行调用，token 会通过回调实时打印
    print("流式输出: ", end="", flush=True)
    response = chain.invoke({"animal": "猫"})
    print(f"\n\n最终响应: {response.content}")


# ============================================================
# 示例5：完整的事件追踪 Handler
# ============================================================

class FullEventTracker(BaseCallbackHandler):
    """完整的事件追踪回调处理程序"""

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> None:
        """聊天模型开始事件"""
        print(f"\n[事件] Chat Model Start")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新 Token 生成事件"""
        print(f"[事件] New Token: {token}", end="", flush=True)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 结束事件"""
        print(f"\n[事件] LLM End")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 错误事件"""
        print(f"[事件] LLM Error: {error}")

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Chain 开始事件"""
        print(f"[事件] Chain Start - inputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chain 结束事件"""
        print(f"[事件] Chain End - outputs: {outputs}")

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Chain 错误事件"""
        print(f"[事件] Chain Error: {error}")

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Tool 开始事件"""
        print(f"[事件] Tool Start - input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Tool 结束事件"""
        print(f"[事件] Tool End - output: {output}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Tool 错误事件"""
        print(f"[事件] Tool Error: {error}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """通用文本事件"""
        print(f"[事件] Text: {text}")


def demo_full_event_tracker():
    """完整事件追踪示例"""
    print("\n" + "=" * 50)
    print("4. 完整事件追踪")
    print("=" * 50)

    from langchain_core.tools import tool

    # 定义一个简单的加法工具
    @tool
    def add(a: int, b: int) -> int:
        """
        Add two numbers.

        Args:
            a: 第一个整数
            b: 第二个整数
        Returns:
            两数之和
        """
        return a + b

    # 创建提示模板
    prompt = ChatPromptTemplate.from_template("计算 {a} + {b} 的结果")

    # 构建链
    chain = prompt | model

    # 使用完整事件追踪
    tracker = FullEventTracker()

    # 通过 config 传递回调处理器
    response = chain.invoke(
        {"a": 5, "b": 3},
        config={"callbacks": [tracker]}
    )
    print(f"\n最终响应: {response.content}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print(f"模型配置: {model_name}")
    print(f"API 地址: {base_url}")
    print("=" * 50)

    # 运行时传递回调（推荐方式，适合临时调试）
    demo_runtime_callback()

    # 构造函数回调（适合持续监控）
    demo_constructor_callback()

    # 自定义流式输出（适合实时展示）
    demo_streaming_callback()

    # 完整事件追踪（适合调试复杂应用）
    demo_full_event_tracker()
