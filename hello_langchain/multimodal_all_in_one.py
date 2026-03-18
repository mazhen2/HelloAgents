"""
LangChain 多模态输入与自定义输出示例
包含：图像输入、JSON/XML输出解析

本文件演示了 LangChain 的以下功能：
1. 多模态数据输入：支持图像 URL 和 base64 编码的图像数据
2. 自定义输出解析：支持 JSON、XML 等格式的结构化输出
"""

import base64
import os
import httpx
from pydantic import BaseModel, Field

# 从 .env 文件加载环境变量（如 API Keys）
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 模型配置 - 从环境变量读取
model_name = os.getenv("LLM_MODEL_ID", "qwen3.5-flash")
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
timeout = int(os.getenv("LLM_TIMEOUT", "60"))

# 验证必要配置
if not api_key:
    raise ValueError("缺少API密钥：请设置 LLM_API_KEY")
if not base_url:
    raise ValueError("缺少服务地址：请设置 LLM_BASE_URL")

# 初始化模型工厂函数
def create_model(**kwargs):
    """创建 ChatOpenAI 模型实例"""
    return ChatOpenAI(
        model=kwargs.pop("model", model_name),
        api_key=api_key,
        base_url=base_url,
        timeout=kwargs.pop("timeout", timeout),
        **kwargs
    )


# ============================================================
# 一、多模态数据输入
# ============================================================
# LangChain 支持多模态模型输入，除了文本外还可以输入图像
# 图像可以通过以下方式传递：
# 1. base64 编码的图像数据（适合本地图像或需要预处理的情况）
# 2. 直接使用图像 URL（适合网络图像）
# 多模态模型（如 GPT-4V/GPT-4O）能够理解图像内容并根据图像回答问题

def demo_image_input():
    """图像输入示例 - 通过 base64 编码读取图片并描述天气"""
    print("\n" + "=" * 50)
    print("1. 图像输入 - 通过字节流读取图片")
    print("=" * 50)

    # 图片 URL（用户提供的图片）
    image_url = "https://tse1.mm.bing.net/th/id/OIP.VifTW6WkhwQO-9WDUNoZ4wHaEK?rs=1&pid=ImgDetMain&o=7&rm=3"

    # 方式1：通过字节流读取图片并转换为 base64 编码
    # 步骤：
    # 1. httpx.get(image_url).content 获取图片的原始字节数据
    # 2. base64.b64encode() 将字节数据编码为 base64 字符串
    # 3. decode("utf-8") 将字节转换为 UTF-8 字符串
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    # 创建包含图像的人类消息
    # HumanMessage 的 content 参数是一个列表，支持多部分内容：
    # - 文本部分：{"type": "text", "text": "..."}
    # - 图像部分：{"type": "image_url", "image_url": {"url": "..."}}
    # 图像 URL 可以是 base64 数据 URI 格式：data:image/jpeg;base64,{base64数据}
    message = HumanMessage(
        content=[
            {"type": "text", "text": "用中文描述这张图片中的天气"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ]
    )

    # 使用工厂函数创建模型实例
    # 阿里百炼的 GPT-4O 兼容模型
    model = create_model(model="qwen-vl-max")
    # 调用模型获取响应
    response = model.invoke([message])
    print(f"结果: {response.content}")


def demo_image_url_direct():
    """直接使用图像URL - 简化版，无需 base64 编码"""
    print("\n" + "=" * 50)
    print("2. 图像输入 - 直接使用URL")
    print("=" * 50)

    # 直接使用图像的 URL 地址（用户提供的图片）
    # 优点：代码更简洁，无需额外处理
    # 缺点：需要模型支持 URL 访问（GPT-4O 支持）
    image_url = "https://tse1.mm.bing.net/th/id/OIP.VifTW6WkhwQO-9WDUNoZ4wHaEK?rs=1&pid=ImgDetMain&o=7&rm=3"

    # 创建消息，图像部分直接使用 URL
    # 注意：当 image_url 的值是 URL 字符串时，会自动从该 URL 下载图像
    message = HumanMessage(
        content=[
            {"type": "text", "text": "用中文描述这张图片中的天气"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    model = create_model()
    response = model.invoke([message])
    print(f"结果: {response.content}")


def demo_multiple_images():
    """多幅图像输入 - 同时输入多张图片进行比较或综合分析"""
    print("\n" + "=" * 50)
    print("3. 图像输入 - 多幅图像")
    print("=" * 50)

    # 定义两张图片的 URL
    # 图片1：威斯康星大学麦迪逊分校的冬季融雪
    image_url1 = "https://img95.699pic.com/photo/32232/3986.jpg_wh300.jpg!/fh/300/quality/90"
    # 图片2：中国早晨的雪景
    image_url2 = "https://pic.nximg.cn/file/20160615/22720944_155324742000_2.jpg"

    # 在 content 列表中添加多个图像
    # LangChain 支持在单条消息中包含多张图片，模型会综合分析所有图片
    # 使用场景：比较图片差异、识别连续画面、结合多张图片信息回答问题
    message = HumanMessage(
        content=[
            {"type": "text", "text": "这两张图片一样的吗？"},
            {"type": "image_url", "image_url": {"url": image_url1}},
            {"type": "image_url", "image_url": {"url": image_url2}},
        ]
    )

    model = create_model()
    response = model.invoke([message])
    print(f"结果: {response.content}")


# ============================================================
# 二、自定义输出：JSON, XML
# ============================================================
# LangChain 提供了多种输出解析器来格式化模型的响应
# 这些解析器可以将模型的自然语言输出转换为结构化格式：
# 1. JsonOutputParser: 输出 JSON 格式，适合需要结构化数据的场景
# 2. XMLOutputParser: 输出 XML 格式，适合需要标记语言的场景
# 
# 使用步骤：
# 1. 定义 Pydantic 模型指定期望的数据结构
# 2. 创建对应的解析器
# 3. 在 PromptTemplate 中通过 partial_variables 注入格式说明
# 4. 构建 LCEL 链：prompt | model | parser
# 5. 调用链获取结构化输出

# 定义数据模型
# 使用 Pydantic 定义数据结构，Field 用于添加字段描述
# 这些描述会被注入到提示词中，引导模型生成符合格式的输出

class Joke(BaseModel):
    """笑话数据模型"""
    # setup: 笑话的设置/铺垫部分
    setup: str = Field(description="设置笑话的问题")
    # punchline: 笑话的抖包袱/答案部分
    punchline: str = Field(description="解决笑话的答案")


def demo_json_output():
    """JSON输出解析 - 将模型输出解析为 JSON 格式"""
    print("\n" + "=" * 50)
    print("5. JSON输出解析")
    print("=" * 50)

    # 初始化模型和解析器
    # temperature=0 使模型输出更确定性，减少随机性
    model = create_model(temperature=0)
    # 创建 JSON 输出解析器，传入 Pydantic 模型指定输出格式
    # 解析器会根据模型自动生成 JSON Schema 并注入到提示词中
    parser = JsonOutputParser(pydantic_object=Joke)

    # 构建提示模板
    # template: 提示词模板，包含用户查询和格式说明的占位符
    # input_variables: 模板中需要用户提供的变量列表
    # partial_variables: 预填充的变量，这里注入格式说明
    #   - format_instructions 由解析器自动生成，告诉模型如何输出 JSON
    prompt = PromptTemplate(
        template="回答用户的查询。\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建 LCEL 链（LangChain Expression Language）
    # | 运算符将组件串联起来，数据按顺序流动：
    # 1. prompt 接收输入，生成完整的提示词
    # 2. model 接收提示词，生成响应
    # 3. parser 接收模型输出，解析为 JSON
    chain = prompt | model | parser

    # 同步调用：invoke 一次性返回完整结果
    # 输入是一个字典，包含 query 变量
    result = chain.invoke({"query": "讲一个关于程序员的笑话"})
    print(f"同步结果: {result}")
    # 返回的是 Pydantic 模型实例，可以直接访问属性
    # 例如: result.setup, result.punchline

    # 流式处理：stream 逐步返回结果
    # 适用于长输出或需要实时展示的场景
    # 流式返回的是解析后的 JSON 片段（字典或字符串）
    print("\n流式输出:")
    for s in chain.stream({"query": "讲一个关于程序员的笑话"}):
        print(s, end="", flush=True)
    print()


def demo_xml_output():
    """XML输出解析 - 将模型输出解析为 XML 格式"""
    print("\n" + "=" * 50)
    print("6. XML输出解析")
    print("=" * 50)

    # 初始化模型和 XML 解析器
    # XMLOutputParser 不需要传入 Pydantic 模型
    # 它会将模型输出解析为 XML 元素的字典结构
    model = create_model(temperature=0)
    parser = XMLOutputParser()

    # 构建提示模板
    # XMLOutputParser 会自动生成 XML 格式说明
    prompt = PromptTemplate(
        template="回答用户的查询。\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建链
    chain = prompt | model | parser

    # 调用链获取 XML 格式的输出
    # 解析结果是一个字典，键是 XML 标签名，值是标签内容
    result = chain.invoke({"query": "给我推荐一些电影，包含演员、电影名称和类型"})
    print(f"结果: {result}")
    # 例如：{'movie': [{'name': '...', 'actor': '...', 'genre': '...'}, ...]}


# ============================================================
# 主函数
# ============================================================
# 演示入口函数，可以选择运行不同的示例

if __name__ == "__main__":
    # 多模态输入示例
    # 注意：图像相关示例需要网络访问，可能需要较长时间
    # 注意：阿里百炼模型需要支持视觉能力，且图像 URL 需要可访问
    # 以下是函数调用说明：
    # - demo_image_input(): 通过 base64 编码读取本地/网络图片并分析
    # - demo_image_url_direct(): 直接使用图片 URL（代码更简洁）
    # - demo_multiple_images(): 同时输入多张图片进行比较
    
    # 先测试图像输入（需要模型支持视觉能力）
    try:
        demo_image_input()
    except Exception as e:
        print(f"图像输入失败: {e}")
    
    try:
        demo_image_url_direct()
    except Exception as e:
        print(f"图像URL输入失败: {e}")
    
    try:
        demo_multiple_images()
    except Exception as e:
        print(f"多图像输入失败: {e}")

    # 自定义输出示例
    # 以下示例演示如何让模型输出结构化格式：
    # - demo_json_output(): 输出 JSON 格式（推荐用于 API 响应）
    # - demo_xml_output(): 输出 XML 格式（适合需要嵌套结构的场景）
    demo_json_output()
    demo_xml_output()
