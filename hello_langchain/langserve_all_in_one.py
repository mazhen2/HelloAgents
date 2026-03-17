"""
LangChain 服务部署与链路监控完整示例

本文件演示了 LangChain 在生产环境中的关键能力:
1. LangServe 服务器 - 将 LangChain 链部署为 REST API
2. LangServe 客户端 - 调用远程 LangChain 服务
3. 链路监控示例 - Verbose、Debug 模式和 LangSmith Tracing

依赖安装:
    pip install langchain-openai langchain-core langserve fastapi uvicorn requests

运行方式:
    # 启动服务器 (在终端1)
    python langserve_all_in_one.py --mode server

    # 客户端调用 (在终端2，需要先启动服务器)
    python langserve_all_in_one.py --mode client

    # Verbose 模式 - 显示重要事件
    python langserve_all_in_one.py --mode verbose

    # Debug 模式 - 显示所有事件
    python langserve_all_in_one.py --mode debug

    # LangSmith 模式 - 启用 LangSmith 跟踪
    python langserve_all_in_one.py --mode langsmith

    # 运行所有演示
    python langserve_all_in_one.py --mode all

注意事项:
    - 运行前需设置 OPENAI_API_KEY 环境变量
    - LangSmith 模式需额外设置 LANGCHAIN_API_KEY
"""

import os
import argparse
import requests

# 尝试加载 .env 文件 (如果存在)
try:
    from dotenv import load_dotenv
    # 尝试从项目根目录加载 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    pass  # 如果没有 dotenv,跳过

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.runnables import RunnableMap
from langserve import add_routes, RemoteRunnable


# ==================== 1. LangServe 服务器 ====================
# LangServe 是 LangChain 的官方服务框架,用于将 LangChain 链部署为可调用的 REST API
# 它基于 FastAPI 构建,自动提供:
#   - /invoke 端点: 同步调用链
#   - /stream 端点: 流式响应
#   - /batch 端点: 批量处理
#   - /input_schema 端点: 获取输入 JSON Schema
#   - /output_schema 端点: 获取输出 JSON Schema
#   - 自动生成 API 文档 (/docs)

def create_server_app():
    """
    创建 LangServe 服务器应用
    
    此函数构建一个完整的 FastAPI 应用,包含:
    1. CORS 中间件 - 允许跨域请求
    2. 聊天模型 - 使用 GPT-4o
    3. 提示模板 - 定义系统和人消息格式
    4. 链 - 将提示、模型和输出解析器组合
    5. 路由 - 使用 add_routes 注册链端点
    
    返回:
        FastAPI: 配置好的 FastAPI 应用实例
    
    API 端点说明:
        POST /chat/invoke - 聊天接口,输入 {"input": "用户消息"}
        POST /chat/stream - 聊天流式接口
        GET  /chat/input_schema - 获取输入模式
        GET  /chat/output_schema - 获取输出模式
        POST /openai/invoke - OpenAI 兼容接口
        POST /story/invoke - 故事生成接口
    """
    # 初始化 FastAPI 应用,配置 API 文档信息
    app = FastAPI(
        title="LangChain 服务器",      # API 标题,显示在 Swagger 文档中
        version="1.0",                  # API 版本号
        description="使用 LangChain 的 Runnable 接口的简单 API 服务器",  # API 描述
    )

    # 设置 CORS 中间件,允许浏览器跨域访问
    # 参数说明:
    #   - allow_origins: 允许的源,["*"] 表示允许所有源
    #   - allow_credentials: 是否允许携带凭证(如 cookies)
    #   - allow_methods: 允许的 HTTP 方法
    #   - allow_headers: 允许的 HTTP 头
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            # 生产环境应限制为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 创��聊天模型实例
    # ChatOpenAI 是 LangChain 对 OpenAI Chat API 的封装
    # 参数说明:
    #   - model: 使用的模型名称,支持 gpt-4o, gpt-4-turbo, gpt-3.5-turbo 等
    #   - temperature: 生成随机性,0-2 之间,越高越有创意
    #   - max_tokens: 最大生成 token 数
    # 创建聊天模型实例
    # 使用阿里云通义千问模型 (兼容 OpenAI API)
    # 从环境变量读取配置
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "qwen3.5-flash"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60"))
    )

    # 创建聊天提示模板
    # ChatPromptTemplate 是 LangChain 的提示模板类
    # 支持多种消息类型:
    #   - "system": 系统消息,定义助手行为
    #   - "human": 用户消息
    #   - "ai": AI 消息
    # 模板中的 {input} 是占位符,在调用时会被替换为实际输入
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个擅长讲笑话的助手。请用简洁有趣的方式回答。"),
        ("human", "{input}"),  # {input} 占位符会被实际用户输入替换
    ])

    # 创建故事提示模板
    # 与聊天模板不同,这里使用 "topic" 作为输入参数
    story_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个擅长写故事的助手。"),
        ("human", "写一个故事，主题是：{topic}")  # {topic} 占位符
    ])

    # 创建 LangChain 链 (Chain)
    # 链使用 | 运算符组合多个组件: prompt | model | parser
    # 数据流向: input -> prompt.format() -> model.invoke() -> parser.parse() -> output
    
    # 聊天链: 提示模板 -> 模型 -> 字符串输出解析器
    # StrOutputParser 将模型输出转换为字符串格式
    chat_chain = chat_prompt | model | StrOutputParser()
    
    # 故事链: 提示模板 -> 模型 (输出原始 Message 对象)
    story_chain = story_prompt | model

    # 使用 add_routes 注册链到 FastAPI
    # add_routes 是 langserve 提供的便捷函数,自动创建 LangChain 兼容的 API 端点
    # 参数说明:
    #   - app: FastAPI 应用实例
    #   - chain: LangChain Runnable 对象
    #   - path: API 路由路径 (会添加 /invoke, /stream 等后缀)
    add_routes(app, chat_chain, path="/chat")
    add_routes(app, model, path="/openai")    # 直接注册模型,支持 OpenAI 兼容接口
    add_routes(app, story_chain, path="/story")

    # 定义根路由,提供服务器基本信息
    @app.get("/")
    def root():
        """根路由处理器,返回欢迎信息和文档链接"""
        return {
            "message": "欢迎使用 LangChain 服务器", 
            "docs": "/docs"  # 指向自动生成的 Swagger 文档
        }

    return app


def run_server(port: int = 8000):
    """
    运行 LangServe 服务器
    
    使用 uvicorn 作为 ASGI 服务器运行 FastAPI 应用
    
    参数:
        port: 服务器监听端口,默认 8000
    
    启动后访问:
        - http://localhost:8000 - 根路由
        - http://localhost:8000/docs - Swagger API 文档
        - http://localhost:8000/chat/invoke - 聊天接口
    
    注意:
        - 服务器会阻塞当前线程,需在独立终端运行
        - 确保 OPENAI_API_KEY 环境变量已设置
    """
    import uvicorn
    app = create_server_app()
    print(f"启动服务器: http://localhost:{port}")
    print(f"文档: http://localhost:{port}/docs")
    # uvicorn.run() 启动 ASGI 服务器
    # host="localhost" 限制为本地访问,生产环境应改为 "0.0.0.0"
    uvicorn.run(app, host="localhost", port=port)


# ==================== 2. LangServe 客户端 ====================
# LangServe 客户端提供两种调用远程链的方式:
# 1. RemoteRunnable: 类似本地 Runnable 的调用方式
# 2. requests: 标准的 HTTP 请求方式

def demo_client():
    """
    演示 LangServe 客户端调用远程服务
    
    展示三种调用 LangServe 服务器的方式:
    1. RemoteRunnable SDK - 最简洁,最接近本地调用
    2. requests 库 - 标准的 HTTP 调用
    3. 流式调用 - 处理流式响应
    
    前置条件:
        服务器必须已启动 (python langserve_all_in_one.py --mode server)
    
    错误处理:
        捕获 requests.exceptions.ConnectionError,提示用户启动服务器
    """
    print("=" * 60)
    print("LangServe 客户端示例")
    print("=" * 60)

    # 服务器基础 URL
    base_url = "http://localhost:8000"

    # 方式1: 使用 RemoteRunnable SDK
    # RemoteRunnable 是 langserve 提供的客户端封装
    # 允许像调用本地链一样调用远程链
    # 特点:
    #   - 自动处理 HTTP 通信
    #   - 支持 .invoke(), .stream(), .batch() 方法
    #   - 返回类型与本地链一致
    print("\n--- 方式1: 使用 RemoteRunnable SDK ---")
    try:
        # 创建远程链客户端,URL 需以 / 结尾
        chat = RemoteRunnable(f"{base_url}/chat/")
        # 直接调用,如同本地链一样
        response = chat.invoke({"input": "给我讲个笑话"})
        print(f"响应: {response}")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器，请先运行 python langserve_all_in_one.py --mode server")
        return

    # 方式2: 使用 requests 库直接调用 HTTP API
    # 适合需要更精细控制的场景
    # 请求格式: POST /{path}/invoke, JSON body 为 {"input": {"input": "..."}}
    print("\n--- 方式2: 使用 requests 库 ---")
    try:
        response = requests.post(
            f"{base_url}/chat/invoke",
            json={"input": {"input": "你好"}}  # 嵌套结构: {"input": {"input": "用户消息"}}
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")

    # 方式3: 流式调用
    # 使用 stream=True 获取 SSE (Server-Sent Events) 流式响应
    # 适用于长文本生成,实现打字机效果
    # 注意: 需要正确解析流式数据格式
    print("\n--- 方式3: 流式调用 ---")
    try:
        response = requests.post(
            f"{base_url}/chat/stream",
            json={"input": {"input": "讲一个关于程序员的笑话"}},
            stream=True  # 启用流式模式
        )
        print("流式响应:")
        # iter_lines() 逐行读取响应,每行是一个 JSON 对象
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")

    # 获取 API 模式信息
    # LangServe 自动提供输入输出模式的 HTTP 端点
    # 可用于:
    #   - 动态生成客户端代码
    #   - 验证输入数据
    #   - API 文档生成
    print("\n--- 获取 API 模式 ---")
    try:
        input_schema = requests.get(f"{base_url}/chat/input_schema")
        output_schema = requests.get(f"{base_url}/chat/output_schema")
        print(f"输入模式: {input_schema.json()}")
        print(f"输出模式: {output_schema.json()}")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")


# ==================== 3. 监控示例 ====================
# LangChain 提供多种调试和监控工具:
# 1. Verbose 模式 - 显示关键事件
# 2. Debug 模式 - 显示所有详细信息
# 3. LangSmith - 云端链路追踪服务

# 定义工具函数
# 工具 (Tools) 是 LangChain Agent 的���心组件
# 使用 @tool 装饰器将 Python 函数转换为 LangChain 工具
# 工具会被Agent调用来执行特定任务

@tool
def search(query: str) -> str:
    """
    搜索工具 - 用于查询实时信息
    
    这是 LangChain Agent 可用的工具之一
    Agent 会根据用户问题自主决定是否调用此工具
    
    参数:
        query: 搜索查询字符串
    
    返回:
        str: 格式化的搜索结果字符串
    
    注意:
        这是一个模拟实现,实际使用时应替换为真实搜索 API
        如 Tavily, SerpAPI, 或 DuckDuckGo 等
    """
    return f"搜索结果: 找到了关于「{query}」的最新信息..."


@tool
def calculate(expression: str) -> str:
    """
    计算工具 - 用于执行数学运算
    
    Agent 可调用此工具进行复杂数学计算
    
    参数:
        expression: 数学表达式字符串,如 "2+2", "(3+5)*2"
    
    返回:
        str: 计算结果或错误信息
    
    警告:
        eval() 有安全风险,生产环境应使用 ast.literal_eval 或专用计算库
    """
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def create_agent_executor():
    """
    创建 Agent 执行器
    
    Agent 是 LangChain 的核心概念 - 能够自主决定使用工具完成任务的 AI 系统
    
    本函数创建的工具型 Agent:
    1. 接收用户输入
    2. 决定是否需要调用工具
    3. 调用工具获取结果
    4. 基于工具结果生成最终回答
    
    返回:
        AgentExecutor: 配置好的 Agent 执行器
    
    create_agent 参数:
        - llm: 语言模型 (ChatOpenAI)
        - tools: 可用工具列表 [search, calculate]
        - system_prompt: 系统提示,定义 Agent 行为
    """
    # 创建语言模型实例
    # 使用阿里云通义千问模型 (兼容 OpenAI API)
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "qwen3.5-flash"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60"))
    )
    
    # 定义可用工具列表
    # Agent 会根据任务自主选择调用哪个工具
    tools = [search, calculate]

    # 使用 create_agent 创建 Agent
    # 这是 LangChain 0.1+ 版本的 API
    # 旧版本使用 AgentExecutor.from_agent_and_tools()
    agent_executor = create_agent(
        llm,
        tools,
        system_prompt="你是一个乐于助人的助手。"  # 指导 Agent 行为的系统提示
    )
    return agent_executor


def run_verbose():
    """
    Verbose 模式运行
    
    Verbose 模式显示 LangChain 执行过程中的重要事件:
    - 组件调用
    - 中间结果
    - 工具调用
    - 重要日志
    
    适用于:
    - 了解链的执行流程
    - 调试特定问题
    - 学习 LangChain 内部机制
    
    设置方式:
    - agent_executor.debug = True
    - 或通过 LANGCHAIN_VERBOSE 环境变量
    """
    print("=" * 60)
    print("Verbose 模式 - 只显示重要事件")
    print("=" * 60)
    print()

    agent_executor = create_agent_executor()
    # 新版本通过 debug 属性控制 Verbose 输出
    # 启用后,执行过程中的关键事件会输出到控制台
    agent_executor.debug = True

    # 执行 Agent
    # invoke 方法接受字典,必须包含 "input" 键
    # Agent 会:
    #   1. 分析用户问题 "搜索2025年电影《碟中谍8》的导演是谁？"
    #   2. 判断需要调用 search 工具
    #   3. 调用工具获取结果
    #   4. 生成最终回答
    result = agent_executor.invoke({
        "input": "搜索2025年电影《碟中谍8》的导演是谁？"
    })

    print(f"\n最终结果: {result}")


def run_debug():
    """
    Debug 模式运行
    
    Debug 模式输出最详细的执行信息:
    - 所有组件调用
    - 完整的请求/响应
    - 中间状态变化
    - 详细错误信息
    
    适用于:
    - 深入调试复杂链
    - 排查问题根因
    - 性能分析
    
    与 Verbose 的区别:
    - Verbose: 只显示重要事件
    - Debug: 显示所有事件,包括详细调试信息
    """
    print("=" * 60)
    print("Debug 模式 - 显示所有事件")
    print("=" * 60)
    print()

    agent_executor = create_agent_executor()
    # 启用 debug 模式,输出完整的调试信息
    agent_executor.debug = True

    result = agent_executor.invoke({
        "input": "搜索2025年电影《碟中谍8》的导演是谁？"
    })

    print(f"\n最终结果: {result}")


def run_langsmith():
    """
    LangSmith 模式运行
    
    LangSmith 是 LangChain 的云端可观测性平台,提供:
    - 链路追踪 - 可视化每一次调用
    - 性能分析 - 识别瓶颈
    - 调试工具 - 快速定位问题
    - 评估系统 - 测试链的质量
    
    前置条件:
    1. 在 https://smith.langchain.com/ 注册账号
    2. 创建 API Key
    3. 设置环境变量:
       - LANGCHAIN_API_KEY: 你的 API Key
       - LANGCHAIN_TRACING_V2: "true"
       - LANGCHAIN_PROJECT: "可选,项目名"
    
    使用方式:
        设置环境变量后,LangChain 自动将追踪数据发送到 LangSmith
        无需修改代码
    
    查看追踪:
        访问 https://smith.langchain.com/ 进入控制台
    """
    print("=" * 60)
    print("LangSmith 模式")
    print("=" * 60)

    # 检查环境变量配置
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    tracing = os.environ.get("LANGCHAIN_TRACING_V2")

    if not api_key:
        print("\n警告: 未设置 LANGCHAIN_API_KEY 环境变量")
        print("请设置以下环境变量:")
        print('  Windows: set LANGCHAIN_API_KEY=your-api-key')
        print('  Linux/Mac: export LANGCHAIN_API_KEY="your-api-key"')
        print('  或在 .env 文件中设置')
        print("\n或在代码中设置:")
        print('  os.environ["LANGCHAIN_API_KEY"] = "your-api-key"')
        print('  os.environ["LANGCHAIN_TRACING_V2"] = "true"')
        print("\n请访问 https://smith.langchain.com/ 注册并获取 API Key\n")
        return

    if tracing != "true":
        print(f"\n当前 LANGCHAIN_TRACING_V2 = {tracing}")
        print("设置为 'true' 以启用 LangSmith 跟踪\n")

    # 创建 Agent 执行器
    # 设置环境变量后,LangChain 会自动配置追踪
    agent_executor = create_agent_executor()

    print("执行代理（跟踪将发送到 LangSmith）...\n")
    result = agent_executor.invoke({
        "input": "搜索2025年电影《碟中谍8》的导演是谁？"
    })

    print(f"最终结果: {result}")
    print("\n请访问 https://smith.langchain.com/ 查看完整的跟踪记录")


# ==================== 主函数 ====================

def main():
    """
    主函数 - 命令行入口点
    
    使用 argparse 解析命令行参数,支持多种运行模式:
    - server: 启动 LangServe 服务器
    - client: 调用远程服务示例
    - verbose: Verbose 调试模式
    - debug: Debug 调试模式
    - langsmith: LangSmith 追踪模式
    - all: 显示帮助信息
    
    命令行参数:
        --mode: 运行模式 (默认 "all")
        --port: 服务器端口 (默认 8000)
    
    示例:
        python langserve_all_in_one.py --mode server --port 8080
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="LangChain 服务部署与链路监控示例"
    )
    # 添加 --mode 参数,限制可选值
    parser.add_argument(
        "--mode",
        choices=["server", "client", "verbose", "debug", "langsmith", "all"],
        default="all",
        help="选择运行模式"
    )
    # 添加 --port 参数,指定服务器端口
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LangChain 服务部署与链路监控")
    print("=" * 60)

    # 根据 mode 参数执行不同逻辑
    if args.mode == "server":
        # 启动 LangServe 服务器
        # 服务器会阻塞,需要在独立终端运行
        run_server(args.port)
    elif args.mode == "client":
        # 演示客户端调用远程服务
        # 需要服务器先启动
        demo_client()
    elif args.mode == "verbose":
        # Verbose 调试模式
        run_verbose()
    elif args.mode == "debug":
        # Debug 调试模式
        run_debug()
    elif args.mode == "langsmith":
        # LangSmith 追踪模式
        run_langsmith()
    elif args.mode == "all":
        # 显示帮助信息和可用模式
        print("\n可用模式:")
        print("  --mode server   : 启动 LangServe 服务器")
        print("  --mode client   : 客户端调用示例")
        print("  --mode verbose  : Verbose 模式")
        print("  --mode debug    : Debug 模式")
        print("  --mode langsmith: LangSmith 跟踪")
        print("\n示例:")
        print("  python langserve_all_in_one.py --mode server")
        print("  python langserve_all_in_one.py --mode client")
        print("  python langserve_all_in_one.py --mode verbose")


# 程序入口点
# 当直接运行此文件时,__name__ == "__main__"
# 当作为模块导入时,__name__ != "__main__"
if __name__ == "__main__":
    main()