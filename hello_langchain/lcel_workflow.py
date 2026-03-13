"""
LangChain 工作流编排示例
文档来源: LangChain 工作流编排：从 LCEL 到可运行对象与流处理

包含:
1. Runnable 接口 - stream/invoke/batch
2. Stream 流式输出 - LLM/Chain/JSON
3. Stream Events 事件流
"""

import asyncio
import os

# 从 .env 文件加载环境变量（如 API Keys）
from dotenv import load_dotenv

# LangChain 核心组件
# ChatOpenAI: 通用的 LLM 接口，支持 OpenAI 兼容 API
from langchain_openai import ChatOpenAI

# 输出解析器：将 LLM 输出转换为特定格式
# StrOutputParser: 将 LLM 输出转换为字符串
# JsonOutputParser: 将 LLM 输出解析为 JSON/Pydantic 对象
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 提示词模板：构建结构化的提示
from langchain_core.prompts import ChatPromptTemplate

# 加载 .env 文件中的环境变量
load_dotenv()

# 检查版本（用于调试兼容性问题）
import langchain_core

print(f"langchain-core 版本: {langchain_core.__version__}")

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

# 验证必要配置
if not api_key:
    raise ValueError("缺少API密钥：请设置 LLM_API_KEY")
if not base_url:
    raise ValueError("缺少服务地址：请设置 LLM_BASE_URL")

# 初始化 LLM 模型实例
# 这是 LangChain 中的 Runnable 对象，可以与其他组件组合成 Chain
model = ChatOpenAI(
    model=model_name,      # 模型标识符
    api_key=api_key,      # API 认证
    base_url=base_url,    # 服务地址
    timeout=timeout,      # 请求超时
)


# =============================================================================
# 一、Runnable 接口示例
# =============================================================================
# Runnable 是 LangChain 的核心接口，所有可执行组件都实现此接口
# 主要方法：
#   - invoke/ainvoke: 同步/异步调用
#   - stream/astream: 同步/异步流式输出
#   - batch/abatch: 同步/异步批量处理

def runnable_interface_demo():
    """Runnable 接口的同步和异步方法
    
    展示 LangChain 中 Runnable 对象的四种基本调用方式：
    1. invoke - 单次同步调用
    2. batch - 批量同步调用
    3. stream - 同步流式输出
    4. 异步版本 (ainvoke, abatch, astream)
    """
    
    # 构建一个简单的 Chain：提示词模板 -> 模型 -> 输出解析器
    # 使用 | 运算符（LCEL 语法）将组件串联
    # 数据流向：{"topic": "xxx"} -> prompt -> model -> parser -> "笑话文本"
    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    parser = StrOutputParser()  # 简单起见，直接输出字符串
    chain = prompt | model | parser

    # invoke - 同步单次调用
    # 输入：字典 {"topic": "xxx"}
    # 输出：字符串（笑话文本）
    print("=== invoke (同步) ===")
    result = chain.invoke({"topic": "鹦鹉"})
    print(result[:100] + "...")

    # batch - 同步批量调用
    # 一次处理多个输入，提高效率
    # 输入：多个字典组成的列表
    # 输出：多个结果组成的列表
    print("\n=== batch (批量) ===")
    results = chain.batch([
        {"topic": "鹦鹉"},
        {"topic": "猫"},
        {"topic": "狗"}
    ])
    for r in results:
        print(f"- {r[:50]}...")

    # stream - 同步流式输出
    # 逐块返回结果，类似打字机效果
    # 每块是字符串的一部分
    print("\n=== stream (流式) ===")
    for chunk in chain.stream({"topic": "鹦鹉"}):
        print(chunk, end='|', flush=True)


async def runnable_interface_async_demo():
    """Runnable 接口的异步方法
    
    异步版本适用于需要高并发的场景，如 Web 服务器
    可以在等待 LLM 响应时处理其他任务
    """
    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    parser = StrOutputParser()
    chain = prompt | model | parser

    # ainvoke - 异步单次调用
    # 使用 await 等待结果
    print("\n=== ainvoke (异步) ===")
    result = await chain.ainvoke({"topic": "鹦鹉"})
    print(result[:100] + "...")

    # abatch - 异步批量调用
    # 并发处理多个请求，比顺序 batch 更快
    print("\n=== abatch (异步批量) ===")
    results = await chain.abatch([
        {"topic": "鹦鹉"},
        {"topic": "猫"}
    ])
    for r in results:
        print(f"- {r[:50]}...")

    # astream - 异步流式输出
    # 异步迭代器，逐块获取结果
    print("\n=== astream (异步流式) ===")
    async for chunk in chain.astream({"topic": "鹦鹉"}):
        print(chunk, end='|', flush=True)


# =============================================================================
# 二、Stream 流式输出
# =============================================================================
# 流式输出允许在完整结果生成前逐步显示内容
# 适用于需要实时反馈的场景，如聊天机器人

def stream_lm_demo():
    """LLM/聊天模型流式输出
    
    直接从模型获取流式响应
    每个 chunk 是一个 AIMessageChunk 对象
    """
    print("\n" + "=" * 50)
    print("=== LLM 流式输出 ===")

    chunks = []
    # model.stream() 返回一个迭代器
    # 每次迭代返回一个 chunk（词块）
    for chunk in model.stream("天空是什么颜色？"):
        chunks.append(chunk)
        print(chunk.content, end='|', flush=True)

    print("\n\n--- 查看单个块 ---")
    # chunk 是 AIMessageChunk 类型，包含 content 等属性
    print(chunks[1])

    print("\n--- 块叠加 ---")
    # 可以使用 + 运算符合并多个 chunk
    # LangChain 重载了 + 操作符
    combined = chunks[0] + chunks[1] + chunks[2] + chunks[3] + chunks[4]
    print(combined)


async def stream_lm_async_demo():
    """LLM/聊天模型异步流式输出
    
    异步版本的流式输出，适用于异步框架如 FastAPI
    """
    print("\n" + "=" * 50)
    print("=== LLM 异步流式输出 ===")

    chunks = []
    # astream 是异步迭代器，使用 async for 遍历
    async for chunk in model.astream("天空是什么颜色？"):
        chunks.append(chunk)
        print(chunk.content, end='|', flush=True)


def stream_chain_demo():
    """Chain 流式输出 - 使用 LCEL 构建链
    
    整个 Chain 都支持流式输出
    数据流：输入 -> prompt -> model -> parser -> 输出块
    """
    print("\n" + "=" * 50)
    print("=== Chain 流式输出 ===")

    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    parser = StrOutputParser()
    chain = prompt | model | parser

    # Chain 的 stream 会逐块输出最终解析后的结果
    for chunk in chain.stream({"topic": "鹦鹉"}):
        print(chunk, end='|', flush=True)


async def stream_json_demo():
    """JSON 流式解析 - 流式传输部分 JSON
    
    JsonOutputParser 支持流式解析：
    - 初始：返回空字典或部分字段
    - 过程中：逐步填充字段值
    - 最终：返回完整解析结果
    
    这使得可以在解析过程中就显示部分结果
    """
    print("\n" + "=" * 50)
    print("=== JSON 流式解析 ===")

    # 使用 Pydantic model 强制 JSON 解析
    # Pydantic 提供类型验证，确保输出符合 schema
    from pydantic import BaseModel

    # 定义国家的数据结构
    class Country(BaseModel):
        name: str           # 国家名称
        population: int     # 人口数量

    # 定义国家列表的结构
    class CountryList(BaseModel):
        countries: list[Country]  # 国家列表

    # 创建 JSON 解析器，绑定到 Pydantic 模型
    # 解析器会自动验证和转换 LLM 输出
    parser = JsonOutputParser(pydantic_object=CountryList)
    
    # 构建 Chain：模型 -> JSON 解析器
    chain = model | parser

    # 关键：添加强制 JSON 格式的提示（使用双引号）
    # LLM 有时返回 Python 字典格式（单引号），需要明确要求 JSON（双引号）
    prompt = """请以标准 JSON 格式返回，不要包含任何编程语言标记（如 ```python 或 ```json）。
JSON 必须使用双引号，不能用单引号。

列出法国和日本的国家及其人口列表。
使用一个带有'countries'作为键的字典，其中包含国家列表。
每个国家都应该有键'name'和'population'"""

    print("\n--- 流式调用 ---")
    # 流式 JSON 解析会逐步返回累积的对象
    # 例如：{} -> {'countries': [{}]} -> {'countries': [{'name': '法国'}]} -> ...
    async for text in chain.astream(prompt):
        print(text)


# =============================================================================
# 三、Stream Events 事件流
# =============================================================================
# astream_events 是 LangChain v2 提供的强大调试/监控工具
# 它会流式输出执行过程中的每一步事件
# 
# 常见事件类型：
# - on_chat_model_start: LLM 开始调用
# - on_chat_model_stream: LLM 输出 token
# - on_chat_model_end: LLM 调用结束
# - on_prompt_start/end: Prompt 处理
# - on_parser_start/end: 解析器工作
#
# 事件数据结构：
# {
#     "event": "事件类型",
#     "name": "组件名称",
#     "data": {
#         "input": {...},   # 输入数据
#         "chunk": {...},   # 输出片段
#         "output": {...}   # 最终输出
#     }
# }

async def stream_events_chat_model():
    """事件流 - 聊天模型事件
    
    演示如何监听模型执行过程中的事件
    version="v2" 是 LangChain v2 的事件格式
    """
    print("\n" + "=" * 50)
    print("=== 事件流 - 聊天模型 ===")

    events = []
    # astream_events 返回异步迭代器
    # 每个事件是一个字典，包含 event, name, data 等字段
    async for event in model.astream_events("hello", version="v2"):
        events.append(event)
        print(f"事件: {event.get('event')}")

    # 查看特定事件的详细信息
    print("\n--- on_chat_model_start ---")
    if len(events) > 3:
        # event: 事件类型
        # name: 组件名称
        # data.input: 传递给模型的输入
        print(f"事件类型: {events[3].get('event')}")
        print(f"组件名: {events[3].get('name')}")
        print(f"输入: {events[3].get('data', {}).get('input')}")

    print("\n--- on_chat_model_stream ---")
    if len(events) > 5:
        # data.chunk: 每次返回的文本片段
        print(f"事件类型: {events[5].get('event')}")
        print(f"输出块: {events[5].get('data', {}).get('chunk')}")

    return events


async def stream_events_chain():
    """事件流 - 整个链的事件
    
    对整个 Chain 使用 astream_events
    可以看到数据在 Chain 各组件间流动的过程
    """
    print("\n" + "=" * 50)
    print("=== 事件流 - 完整链 ===")

    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    parser = StrOutputParser()
    chain = prompt | model | parser

    # 遍历整个 Chain 的事件流
    # 可以看到：prompt -> model -> parser 的完整流程
    async for event in chain.astream_events({"topic": "鹦鹉"}, version="v2"):
        print(f"[{event.get('event')}] {event.get('name', 'N/A')}")


async def stream_events_with_details():
    """事件流 - 查看详细输入输出
    
    过滤特定类型的事件，只关注感兴趣的部分
    """
    print("\n" + "=" * 50)
    print("=== 事件流 - 详细输入输出 ===")

    async for event in model.astream_events("你好", version="v2"):
        event_type = event.get("event")
        
        # on_chat_model_start: 模型开始处理输入
        if event_type == "on_chat_model_start":
            print(f"START - 输入: {event.get('data', {}).get('input')}")
        
        # on_chat_model_stream: 模型输出 token
        elif event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            print(f"STREAM - 输出: {chunk.content if chunk else 'N/A'}")
        
        # on_chat_model_end: 模型完成处理
        elif event_type == "on_chat_model_end":
            print(f"END - 完成")


# =============================================================================
# 四、输入输出模式
# =============================================================================
# LangChain 的 Runnable 对象都有 input_schema 和 output_schema
# 用于描述期望的输入格式和输出格式
# 基于 Pydantic 的 JSON Schema

def input_output_schema_demo():
    """检查 Runnable 的输入输出模式
    
    input_schema: 描述输入数据的结构
    output_schema: 描述输出数据的结构
    
    用途：
    - 验证输入/输出格式
    - 自动生成 API 文档
    - 与其他组件类型检查
    """
    print("\n" + "=" * 50)
    print("=== 输入输出模式 ===")

    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    parser = StrOutputParser()
    chain = prompt | model | parser

    # 查看输入模式
    # 对于 chain，输入是 {"topic": "xxx"}
    print("\n--- 输入模式 (input_schema) ---")
    print(chain.input_schema.schema())

    # 查看输出模式
    # 对于这个 chain，输出是字符串
    print("\n--- 输出模式 (output_schema) ---")
    print(chain.output_schema.schema())


# =============================================================================
# 主函数 - 运行示例
# =============================================================================

def main():
    print("LangChain 工作流编排示例")
    print("=" * 50)

    # # 一、Runnable 接口
    # runnable_interface_demo()

    # # 异步 Runnable 接口
    # asyncio.run(runnable_interface_async_demo())

    # 二、Stream 流式输出
    # stream_lm_demo()
    # asyncio.run(stream_lm_async_demo())
    # stream_chain_demo()
    # asyncio.run(stream_json_demo())

    # # 三、Stream Events 事件流
    asyncio.run(stream_events_chat_model())
    # asyncio.run(stream_events_chain())
    # asyncio.run(stream_events_with_details())

    # # 四、输入输出模式
    # input_output_schema_demo()

    print("\n" + "=" * 50)
    print("示例完成!")


if __name__ == "__main__":
    main()
