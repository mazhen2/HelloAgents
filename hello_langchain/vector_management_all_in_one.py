from __future__ import annotations

import os  # 操作系统接口，用于环境变量读取

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv

    # __file__ 是 hello_langchain/task/vector_management_all_in_one.py
    # 需要跳两级目录才能到达项目根目录
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(env_path)

    # 确保 langchain_openai 兼容的环境变量
    # 如果设置了 LLM_API_KEY，也同步到 OPENAI_API_KEY
    if os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")
except ImportError:
    pass

# 标准库导入
import argparse  # 命令行参数解析
import shutil  # 高级文件操作（如删除目录树）
import sys  # Python 运行时接口
import textwrap  # 文本格式化工具
from dataclasses import dataclass  # 数据类装饰器
from pathlib import Path  # 路径对象封装
from typing import Any, Iterable, Optional  # 类型提示
from langchain_openai import ChatOpenAI

# ==================== 公共组件 ====================

# 阿里云百炼 dashscope API
DASHSCOPE_API_KEY = "sk-52254e84633242c9ae4383c7716486f3"

# 创建 Embeddings 对象（使用阿里云百炼）
from langchain_openai import OpenAIEmbeddings

# 创建 embeddings 对象 - 使用阿里云 text-embedding-v4 模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
)

# 创建聊天模型实例
# 使用阿里云通义千问模型
chat = ChatOpenAI(
    model="qwen3.5-flash",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60
)

# ============================================================================
# 全局常量定义
# ============================================================================

# 获取当前脚本所在目录的绝对路径
# __file__ 是当前脚本的文件路径
# .resolve() 将相对路径转换为绝对路径
# .parent 获取父目录（即 hello_langchain/）
ROOT = Path(__file__).resolve().parent

# Chroma 向量数据库的默认持久化存储目录
# 如果不指定 --persist-dir，将使用 hello_langchain/chroma_db
DEFAULT_PERSIST_DIR = ROOT / "chroma_db"


def _print_block(title: str, body: str) -> None:
    """
    打印格式化的标题和内容块

    用于在控制台输出时添加视觉分隔，使输出更清晰易读。

    参数:
        title: 标题文字，将显示在顶部分隔线下方
        body: 正文内容，将显示在标题下方，底部分隔线上方
    """
    # 生成长度为 70 的分隔线
    line = "=" * 70
    # 打印空行 + 分隔线 + 标题 + 分隔线
    print(f"\n{line}\n{title}\n{line}")
    # 使用 textwrap.dedent 去除 body 的公共缩进，然后打印
    # .strip() 去除首尾空白
    print(textwrap.dedent(body).strip() + "\n")


def _optional_import(module: str, pip_hint: str) -> Optional[Any]:
    """
    尝试动态导入模块，处理可选依赖

    这是一个"软依赖"处理机制的辅助函数。
    当某个功能依赖可选的第三方库时，使用此函数可以：
    - 如果导入成功，返回模块对象
    - 如果导入失败，打印友好的错误提示（包含安装命令），返回 None

    参数:
        module: 要导入的模块名称（如 "redis"、"chromadb"）
        pip_hint: 安装提示，包含 pip install 命令

    返回:
        成功导入时返回模块对象，失败时返回 None
    """
    try:
        # 尝试导入模块
        # fromlist=["*"] 确保导入模块的顶层命名空间
        return __import__(module, fromlist=["*"])
    except Exception as e:
        # 导入失败时，打印友好的错误提示
        _print_block(
            f"缺少依赖：{module}",
            f"""
            无法导入 `{module}`：{e}
            你可以安装：
              {pip_hint}
            """,
        )
        return None


def _require_env(name: str) -> bool:
    """
    检查环境变量是否已设置

    某些功能需要特定的环境变量才能运行（如 API keys）。
    此函数检查环境变量是否存在，不存在时打印设置提示。

    参数:
        name: 环境变量名称（如 "OPENAI_API_KEY"）

    返回:
        环境变量已设置时返回 True，未设置时返回 False 并打印提示
    """
    # 检查环境变量是否存在
    if os.environ.get(name):
        return True

    # 环境变量未设置，打印提示
    _print_block(
        f"缺少环境变量：{name}",
        f"""
        当前未设置 `{name}`，对应示例可能无法运行。
        Windows PowerShell 设置示例：
          $env:{name} = "..."
        """,
    )
    return False


def _sample_documents() -> list[dict[str, Any]]:
    """
    返回一个"伪文档列表"，作为向量数据库的测试数据

    为了保持单文件可运行，这里不依赖外部文件，而是直接定义测试文档。
    这些文档涵盖了不同主题，用于演示向量检索的效果。

    返回:
        包含字典的列表，每个字典有 "page_content" 和 "metadata" 两个键
    """
    # 测试文档内容
    texts = [
        "Next 公司由史蒂夫·乔布斯在 1985 年创立。",
        "NeXT（后写作 Next）是一家计算机公司，后来被苹果收购。",
        "余弦相似度用于衡量向量夹角的相似程度。",
        "Chroma 是轻量易用的开源向量数据库，适合原型验证。",
        "Qdrant 提供高性能相似性搜索与丰富的过滤能力。",
    ]
    # 转换为字典格式，添加统一的 metadata
    return [{"page_content": t, "metadata": {"source": "builtin"}} for t in texts]


def _docs_to_langchain_documents(raw_docs: list[dict[str, Any]]):
    """
    将自定义文档格式转换为 LangChain 的 Document 对象

    LangChain 的向量数据库需要使用 langchain_core.documents.Document 对象。
    此函数完成格式转换。

    参数:
        raw_docs: 原始文档列表，每个文档是包含 "page_content" 和 "metadata" 的字典

    返回:
        LangChain Document 对象列表，转换失败时返回 None
    """
    # 尝试导入 langchain_core.documents
    # 这是软依赖处理的另一个例子
    langchain_core_documents = _optional_import(
        "langchain_core.documents",
        "pip install -U langchain-core",
    )
    if langchain_core_documents is None:
        return None

    # 获取 Document 类
    Document = getattr(langchain_core_documents, "Document")

    # 转换文档格式
    return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw_docs]


def demo_milvus() -> None:
    """
    演示 Milvus 向量数据库的使用

    Milvus 是开源的向量数据库，具有：
    - 支持十亿级向量规模
    - 多种索引类型（IVF、HNSW、DiskANN 等）
    - 分布式架构支持
    - 丰富的过滤表达式

    本示例演示如何：
    1. 连接本地 Milvus 服务
    2. 从文档创建向量集合
    3. 执行相似度搜索

    注意：
    - Milvus 通常通过 docker compose 启动
    - 默认端口：19531 (gRPC)
    - 需要 pymilvus 客户端库
    """
    _print_block(
        "Milvus：写入与检索（需要本地 Milvus 服务）",
        """
        文档示例使用 pymilvus connections.connect(host, port)。

        依赖（按文档）：
          pip install -U pymilvus langchain-community langchain-openai

        注意：
        - 该示例依赖 Milvus 服务（通常通过 docker compose 启动）。
        """,
    )

    # 检查环境变量（使用阿里云通义千问）
    if not _require_env("LLM_API_KEY"):
        return

    # 动态导入依赖
    pymilvus = _optional_import("pymilvus", "pip install -U pymilvus")
    if pymilvus is None:
        return

    # 获取 connections 模块（用于连接 Milvus 服务）
    connections = getattr(pymilvus, "connections")
    Collection = getattr(pymilvus, "Collection")
    CollectionSchema = getattr(pymilvus, "CollectionSchema")
    DataType = getattr(pymilvus, "DataType")
    FieldSchema = getattr(pymilvus, "FieldSchema")

    # 连接 Milvus 服务
    # 用户配置：Milvus 服务地址 localhost:19530
    # 如需修改为其他地址，请修改以下 host 和 port 参数
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"

    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"已连接到 Milvus 服务：{MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"无法连接到 Milvus 服务（请先启动 Milvus 容器）：{e}")
        return

    # 创建 embeddings 对象（使用阿里云百炼）
    from langchain_openai import OpenAIEmbeddings

    local_embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key="sk-52254e84633242c9ae4383c7716486f3",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
    )

    # 获取测试文档
    raw_docs = _sample_documents()
    if raw_docs is None:
        return

    # 提取文本内容
    texts = [doc["page_content"] for doc in raw_docs]

    # 计算 embeddings
    try:
        print("正在计算 embeddings...")
        vectors = local_embeddings.embed_documents(texts)
        print(f"成功计算 {len(vectors)} 个向量，每个向量维度: {len(vectors[0])}")
    except Exception as e:
        print(f"无法计算 embeddings：{e}")
        return

    # 定义查询
    query = "Qdrant 有什么优势？"

    try:
        # 创建 Collection
        collection_name = "LangChain_Demo"

        # 检查 collection 是否存在，如果存在则删除
        try:
            collection = Collection(collection_name)
            collection.drop()
            print(f"已删除已存在的 collection: {collection_name}")
        except Exception:
            pass

        # 创建 collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(vectors[0])),
        ]
        schema = CollectionSchema(fields=fields, description="LangChain demo collection")
        collection = Collection(name=collection_name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

        # 插入数据
        entities = [
            texts,
            vectors,
        ]
        insert_result = collection.insert(entities)
        collection.flush()
        print(f"成功写入 {len(texts)} 条数据到 Milvus")

        # 执行相似度搜索
        query_vector = local_embeddings.embed_query(query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text"],
        )

        print(f"\n查询：{query}")
        for i, hits in enumerate(results, start=1):
            for hit in hits:
                print(f"[Top {i}] {hit.entity.get('text')}")

    except Exception as e:
        print(f"Milvus 操作失败：{e}")
        import traceback
        traceback.print_exc()


# 程序入口点
if __name__ == "__main__":
    demo_milvus()
