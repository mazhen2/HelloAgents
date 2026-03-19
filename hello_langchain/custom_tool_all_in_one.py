"""
LangChain 自定义工具调用示例
包含：@tool装饰器、StructuredTool、错误处理、内置工具包使用

本文件演示了 LangChain 中创建和使用自定义工具的多种方式，包括：
1. 使用 @tool 装饰器创建同步和异步工具
2. 使用 StructuredTool.from_function 方法创建工具
3. 使用 Pydantic 模型定义工具参数模式
4. 工具的错误处理机制
5. 加载和使用内置工具包（如 Wikipedia、SQL Agent 等）
"""

import os

# 从 .env 文件加载环境变量（如 API Keys）
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool, ToolException
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.load_tools import load_tools
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
    model=model_name,      # 模型标识符
    api_key=api_key,      # API 认证
    base_url=base_url,    # 服务地址
    timeout=timeout,      # 请求超时
)


# ============================================================
# 一、自定义工具的三种创建方式
# ============================================================
# LangChain 提供了多种创建工具的方式，下面分别介绍：
# 1. @tool 装饰器：最简洁的方式，适合简单工具
# 2. StructuredTool.from_function：更灵活，适合复杂场景

# ------------------------------------------------------------
# 方式1：@tool 装饰器
# ------------------------------------------------------------
# @tool 装饰器是创建工具最简单的方式，只需在普通函数上添加装饰器即可。
# LangChain 会自动从函数的文档字符串和类型注解中提取工具的名称、描述和参数信息。

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    
    参数说明:
        a (int): 第一个乘数
        b (int): 第二个乘数
    返回:
        int: 两个数的乘积
    """
    return a * b


@tool
async def async_multiply(a: int, b: int) -> int:
    """
    Multiply two numbers asynchronously.
    
    这是一个异步工具示例，适用于需要并发执行的场景。
    异步工具在 agent 运行时可以获得更好的性能。
    
    参数说明:
        a (int): 第一个乘数
        b (int): 第二个乘数
    返回:
        int: 两个数的乘积
    """
    return a * b


# ------------------------------------------------------------
# 使用 Pydantic 模型自定义参数
# ------------------------------------------------------------
# 当需要更精细的参数控制时，可以使用 Pydantic 模型（BaseModel）来定义参数模式。
# 这允许你：
# 1. 为每个参数添加详细的描述（用于 LLM 理解参数含义）
# 2. 设置参数的默认值
# 3. 添加参数验证逻辑
# 4. 提供更清晰的参数名称（可覆盖函数参数名）
class CalculatorInput(BaseModel):
    """
    计算器工具的参数模式
    
    使用 Pydantic 模型定义工具参数，可以为每个参数提供：
    - description: 描述参数的作用，帮助 LLM 理解如何填充参数
    - default: 参数的默认值（可选）
    - ge/le/gt/lt: 数值边界约束（可选）
    """
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply_with_schema(a: int, b: int) -> int:
    """
    Multiply two numbers.
    
    此工具使用自定义的 Pydantic 参数模式，允许更精细的参数控制。
    当 return_direct=True 时，工具的返回值会直接返回给用户，
    而不是继续传递给后续的 agent 处理流程。
    
    参数说明:
        a (int): 第一个乘数
        b (int): 第二个乘数
    返回:
        int: 两个数的乘积
    """
    return a * b


def demo_tool_decorator():
    """
    @tool 装饰器示例
    
    演示 @tool 装饰器的基本用法，包括：
    - 查看工具的元信息（名称、描述、参数）
    - 调用工具执行计算
    - 使用自定义参数模式的工具
    """
    print("\n" + "=" * 50)
    print("1. @tool 装饰器")
    print("=" * 50)

    # 查看工具信息
    # @tool 装饰器会自动从函数签名和文档字符串中提取信息
    print(f"工具名称: {multiply.name}")  # 默认使用函数名
    print(f"工具描述: {multiply.description}")  # 从文档字符串第一行提取
    print(f"工具参数: {multiply.args}")  # 从类型注解中提取参数信息

    # 调用工具
    # 使用 .invoke() 方法执行工具，传入参数字典
    result = multiply.invoke({"a": 3, "b": 4})
    print(f"调用结果: {result}")

    # 自定义参数的工具
    # 使用自定义 Pydantic 模式的工具仍然通过相同的接口调用
    result2 = multiply_with_schema.invoke({"a": 5, "b": 6})
    print(f"自定义参数工具结果: {result2}")
    # 注意：当 return_direct=True 时，结果会直接返回给用户


# ============================================================
# 方式2：StructuredTool.from_function
# ============================================================
# StructuredTool.from_function 提供了更灵活的工具创建方式，
# 适合需要自定义工具名称、描述或需要复用现有函数的场景。

def multiply_sync(a: int, b: int) -> int:
    """
    Multiply two numbers.
    
    同步乘法函数，可被 StructuredTool 包装成工具。
    
    参数说明:
        a (int): 第一个乘数
        b (int): 第二个乘数
    返回:
        int: 两个数的乘积
    """
    return a * b


async def multiply_async(a: int, b: int) -> int:
    """
    Multiply two numbers asynchronously.
    
    异步乘法函数，适用于高并发场景。
    
    参数说明:
        a (int): 第一个乘数
        b (int): 第二个乘数
    返回:
        int: 两个数的乘积
    """
    return a * b


def demo_structured_tool():
    """
    StructuredTool 示例
    
    演示 StructuredTool.from_function 的两种用法：
    1. 基础用法：直接包装函数
    2. 自定义用法：指定名称、描述、参数模式等
    """
    print("\n" + "=" * 50)
    print("2. StructuredTool.from_function")
    print("=" * 50)

    # 基础用法
    # StructuredTool.from_function 会自动从函数提取信息
    # name: 工具名称（默认使用函数名）
    # description: 工具描述（从文档字符串提取）
    # args: 工具参数（从类型注解提取）
    calculator = StructuredTool.from_function(func=multiply_sync)
    print(f"工具名称: {calculator.name}")
    print(f"工具描述: {calculator.description}")

    result = calculator.invoke({"a": 7, "b": 8})
    print(f"调用结果: {result}")

    # 自定义参数
    # 通过参数可以完全自定义工具的行为
    calculator_custom = StructuredTool.from_function(
        func=multiply_sync,  # 要包装的函数
        name="multiply_numbers",  # 自定义工具名称
        description="Multiply two numbers.",  # 自定义工具描述
        args_schema=CalculatorInput,  # 使用 Pydantic 模型定义参数
        return_direct=True,  # 直接返回结果，不经过 agent 处理
    )
    print(f"\n自定义工具名称: {calculator_custom.name}")
    print(f"return_direct: True")
    # return_direct=True 的使用场景：
    # - 简单的计算工具，返回值不需要进一步处理
    # - 信息查询工具，直接返回查询结果
    # - 工具本身已经包含了完整的信息，不需要 agent 继续处理


# ============================================================
# 错误处理
# ============================================================
# LangChain 提供了灵活的错误处理机制，允许你：
# 1. 使用默认的错误处理（返回错误信息）
# 2. 自定义错误处理函数（返回更有用的错误信息）
# 3. 根据错误类型进行不同的处理

def get_weather(city: str) -> str:
    """
    获取指定城市的天气。
    
    此函数演示如何抛出 ToolException 异常。
    当工具执行失败时，应该抛出 ToolException 来通知 agent。
    
    参数说明:
        city (str): 城市名称
    返回:
        str: 天气信息
    异常:
        ToolException: 当城市不存在时抛出
    """
    raise ToolException(f"错误: 没有名为{city}的城市。")


def handle_error(error: ToolException) -> str:
    """
    自定义错误处理函数
    
    此函数接收 ToolException 异常并返回用户友好的错误消息。
    可以用于：
    - 格式化错误信息
    - 记录错误日志
    - 返回替代建议
    
    参数说明:
        error (ToolException): 捕获的异常对象
    返回:
        str: 格式化的错误消息
    """
    return f"工具执行期间发生以下错误: {error.args[0]}"


def demo_error_handling():
    """
    错误处理示例
    
    演示 LangChain 的两种错误处理方式：
    1. 使用 handle_tool_error=True 启用默认错误处理
    2. 传入自定义错误处理函数
    """
    print("\n" + "=" * 50)
    print("3. 错误处理")
    print("=" * 50)

    # 方式1：使用 handle_tool_error=True
    # 当设置为 True 时，工具执行出错会返回错误信息字符串
    get_weather_tool = StructuredTool.from_function(
        func=get_weather,  # 可能抛出异常的函数
        handle_tool_error=True,  # 启用默认错误处理
    )

    result = get_weather_tool.invoke({"city": "foobar"})
    print(f"默认错误处理结果: {result}")
    # 默认行为：返回类似 "Error: {error_message}" 的字符串

    # 方式2：自定义错误处理
    # 传入一个函数来处理错误，可以返回更友好的消息或替代值
    get_weather_tool_custom = StructuredTool.from_function(
        func=get_weather,  # 可能抛出异常的函数
        handle_tool_error=handle_error,  # 自定义错误处理函数
    )

    result2 = get_weather_tool_custom.invoke({"city": "beijing"})
    print(f"自定义错误处理结果: {result2}")
    # 自定义处理：返回 "工具执行期间发生以下错误: {error_message}"


# ============================================================
# 二、调用内置工具包和拓展工具
# ============================================================
# LangChain 提供了丰富的内置工具，可以直接加载使用：
# - Wikipedia: 维基百科查询
# - LLM Math: 数学计算（使用 LLM 进行复杂数学运算）
# - SQL Database: SQL 数据库查询
# - Search: 网络搜索
# 还可以创建自定义的基于现有 API 的工具

def demo_wikipedia_tool():
    """
    Wikipedia 工具示例

    演示如何使用 WikipediaQueryRun 工具进行维基百科查询。
    需要安装 langchain-community 和 wikipedia Python 包。

    WikipediaAPIWrapper 参数说明:
        top_k_results: 返回的最大结果数量
        doc_content_chars_max: 文档内容的最大字符数
    """
    print("\n" + "=" * 50)
    print("4. Wikipedia 工具")
    print("=" * 50)

    try:
        # 创建 Wikipedia API 包装器
        # top_k_results: 指定返回的结果数量
        # doc_content_chars_max: 限制每个结果文档的字符数，避免返回过长内容
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

        # 创建 Wikipedia 查询工具
        tool = WikipediaQueryRun(api_wrapper=api_wrapper)

        # 执行查询
        result = tool.invoke({"query": "langchain"})
        print(f"Wikipedia 查询结果: {result[:200]}...")
    except ImportError as e:
        print(f"缺少依赖包，请运行: pip install wikipedia")
        print(f"错误详情: {e}")
    except Exception as e:
        print(f"Wikipedia 查询失败: {e}")
        print("注意: 某些地区可能无法访问 Wikipedia API")


def demo_customized_wikipedia_tool():
    """
    自定义 Wikipedia 工具示例

    演示如何自定义 Wikipedia 工具的配置：
    1. 自定义工具名称和描述
    2. 使用自定义的 Pydantic 参数模式
    3. 设置 return_direct=True 直接返回结果
    """
    print("\n" + "=" * 50)
    print("5. 自定义 Wikipedia 工具")
    print("=" * 50)

    try:
        # 定义自定义参数模式
        # description 应包含对 LLM 的指导，帮助其正确构造参数
        class WikiInputs(BaseModel):
            query: str = Field(
                description="query to look up in wikipedia, should be 3 or less words"
            )

        # 创建 API 包装器
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

        # 创建自定义配置的 Wikipedia 工具
        tool = WikipediaQueryRun(
            name="wiki-tool",  # 自定义工具名称
            description="look up things in wikipedia",  # 自定义描述
            args_schema=WikiInputs,  # 自定义参数模式
            api_wrapper=api_wrapper,  # 传入 API 包装器
            return_direct=True,  # 直接返回结果
        )

        print(f"自定义工具名称: {tool.name}")
        print(f"自定义工具描述: {tool.description}")
        result = tool.invoke({"query": "python language"})
        print(f"查询结果: {result[:200]}...")
    except ImportError as e:
        print(f"缺少依赖包，请运行: pip install wikipedia")
        print(f"错误详情: {e}")


def demo_load_tools():
    """
    load_tools 加载内置工具包

    load_tools 是批量加载预置工具的便捷方式。
    支持的工具列表包括：
    - "wikipedia": 维基百科查询
    - "llm-math": 数学计算
    - "search": 网络搜索
    - "python-repl": Python 解释器
    - "requests": HTTP 请求工具

    注意：某些工具（如 llm-math）需要传入 LLM 实例。
    """
    print("\n" + "=" * 50)
    print("6. load_tools 加载内置工具包")
    print("=" * 50)

    try:
        # 加载多个工具只需在列表中指定工具名称
        # llm-math 工具需要 LLM 来执行数学计算
        tools = load_tools(["wikipedia", "llm-math"], llm=model)
        print(f"加载的工具: {[t.name for t in tools]}")

        # 测试 llm-math 工具：计算 2 的 10 次方
        print("\n测试 llm-math 工具:")
        math_tool = tools[1]  # llm-math
        result = math_tool.invoke("2 的 10 次方是多少？")
        print(f"计算结果: {result}")
    except ImportError as e:
        print(f"缺少依赖包，请运行: pip install wikipedia")
        print(f"错误详情: {e}")
    except Exception as e:
        print(f"加载工具时出错: {e}")


def demo_agent_with_tools():
    """
    Agent 绑定工具示例

    演示如何将工具绑定到 LLM 模型，让 Agent 能够自动调用工具。
    这是 ReAct (Reasoning + Acting) 模式的实现。
    """
    print("\n" + "=" * 50)
    print("8. Agent 绑定工具")
    print("=" * 50)

    # 定义一个简单的工具列表
    tools = [multiply, multiply_with_schema]

    # 将工具绑定到模型
    # 模型现在知道有哪些工具可用，可以自动决定何时调用
    model_with_tools = model.bind_tools(tools)

    # 构造提示，让模型决定是否调用工具
    # 注意：由于我们使用的是通用对话模型（非 agent 模型），
    # 这里只是演示工具绑定的基本机制，实际 agent 行为需要使用专门的 agent 类
    messages = [
        {"role": "user", "content": "请帮我计算 15 乘以 23 等于多少？"}
    ]

    try:
        # 调用模型，模型会根据提示决定是否调用工具
        response = model_with_tools.invoke(messages)

        print(f"模型响应类型: {type(response).__name__}")
        print(f"模型输出: {response}")

        # 如果模型决定调用工具，响应中会包含 tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\n模型决定调用工具:")
            for tool_call in response.tool_calls:
                print(f"  - 工具名称: {tool_call['name']}")
                print(f"  - 工具参数: {tool_call['args']}")

    except Exception as e:
        print(f"Agent 调用出错: {e}")
        print("注意: 确保 .env 文件中正确配置了 LLM_API_KEY 和 LLM_BASE_URL")


def demo_sql_agent():
    """
    SQL Agent 工具包

    演示如何使用 create_sql_agent 创建 SQL 查询 Agent。
    这种 Agent 可以：
    1. 理解自然语言问题
    2. 自动生成 SQL 查询
    3. 执行查询并返回结果
    4. 处理数据库错误

    使用场景：
    - 数据库管理员查询数据
    - 数据分析师获取统计信息
    - 业务人员自助查询
    """
    print("\n" + "=" * 50)
    print("9. SQL Agent 工具包 (MySQL)")
    print("=" * 50)

    # 数据库配置
    db_config = {
        "host": "127.0.0.1",
        "port": 3307,
        "database": "keyun",
        "user": "root",
        "password": "123456",
    }

    # 构建 MySQL 连接 URL
    # 格式: mysql+mysqlconnector://user:password@host:port/database
    db_url = (
        f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    try:
        # 连接数据库
        print(f"正在连接 MySQL 数据库: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        db = SQLDatabase.from_uri(db_url)

        # 获取数据库表信息
        print("\n数据库表结构:")
        table_info = db.get_table_info()
        print(table_info[:500] + "..." if len(table_info) > 500 else table_info)

        # 创建 SQL Agent
        print("\n创建 SQL Agent...")
        agent = create_sql_agent(
            llm=model,           # 使用配置好的模型
            db=db,                # 数据库连接
            verbose=True,         # 打印详细执行过程
            agent_type="openai-tools",  # Agent 类型
        )

        # 使用自然语言查询数据库
        print("\n" + "-" * 50)
        print("执行自然语言查询:")
        print("-" * 50)

        # 示例查询1: 获取表列表
        result1 = agent.invoke("这个数据库中有哪些表？")
        print(f"\n查询结果1:\n{result1}")

        # 示例查询2: 统计表数量
        result2 = agent.invoke("一共有多少张表？")
        print(f"\n查询结果2:\n{result2}")

    except ImportError as e:
        print(f"缺少依赖包，请运行: pip install mysql-connector-python")
        print(f"错误详情: {e}")
    except Exception as e:
        print(f"SQL Agent 执行出错: {e}")
        print("\n可能的原因:")
        print("1. MySQL 服务未启动")
        print("2. 数据库连接配置错误")
        print("3. 数据库中不存在 'keyun' 数据库")


# ============================================================
# 主函数
# ============================================================
# 运行本文件将依次执行所有示例，展示 LangChain 工具的各个方面

if __name__ == "__main__":
    print(f"模型配置: {model_name}")
    print(f"API 地址: {base_url}")
    print("=" * 50)

    # ---------------------------------------------------------
    # 第一部分：自定义工具示例
    # ---------------------------------------------------------
    # 演示 @tool 装饰器、StructuredTool 和错误处理的基本用法
    demo_tool_decorator()       # @tool 装饰器基础用法
    demo_structured_tool()       # StructuredTool.from_function 用法
    demo_error_handling()        # 工具错误处理机制

    # ---------------------------------------------------------
    # 第二部分：内置工具示例
    # ---------------------------------------------------------
    # 演示如何使用和自定义 LangChain 内置的工具
    demo_wikipedia_tool()               # Wikipedia 查询工具
    demo_customized_wikipedia_tool()    # 自定义 Wikipedia 工具
    demo_load_tools()                   # 批量加载内置工具
    demo_agent_with_tools()             # Agent 绑定工具
    demo_sql_agent()                    # SQL 数据库 Agent
