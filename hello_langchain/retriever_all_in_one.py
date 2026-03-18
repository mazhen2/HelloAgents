"""
LangChain FAISS 向量数据库 - 单文件示例

本脚本演示 FAISS 向量数据库的使用：
1. 构建 FAISS 索引
2. 相似度搜索
3. 作为检索器使用
4. 带分数检索
5. 持久化保存/加载

依赖：
  pip install -U langchain-community faiss-cpu langchain-openai langchain-text-splitters

运行：
  python retriever_all_in_one.py
"""

# ============================================================
# 导入声明
# ============================================================
# from __future__ import annotations: 启用 PEP 563 延迟注解求值
# 这样可以在函数定义中使用自引用类型（如 "FAISS" 作为类型提示）
from __future__ import annotations

# argparse: 用于解析命令行参数（如 --persist-dir, --reset）
import argparse

# shutil: 用于高级文件操作，如递归删除目录树（rmtree）
import shutil

# textwrap: 用于文本格式化，特别是处理多行字符串的缩进
import textwrap

# pathlib.Path: 面向对象的文件系统路径操作，比 os.path 更直观
from pathlib import Path

# typing: 提供类型注解支持
# Any: 动态类型，表示任意类型
# Optional: 表示可以是某类型或 None
from typing import Any, Optional

# ============================================================
# 全局配置
# ============================================================

# 阿里云百炼 dashscope API 密钥
# 注意：实际项目中建议使用环境变量存储，不要硬编码在代码中
# 设置方式：Windows PowerShell: $env:DASHSCOPE_API_KEY="your-key"
#          Linux/macOS: export DASHSCOPE_API_KEY="your-key"
DASHSCOPE_API_KEY = "sk-52254e84633242c9ae4383c7716486f3"

# ROOT: 获取脚本所在目录的绝对路径
# Path(__file__).resolve() 获取当前脚本文件的绝对路径
# .parent 获取父目录（即脚本所在目录）
ROOT = Path(__file__).resolve().parent

# DEFAULT_FAISS_DIR: FAISS 索引默认持久化目录
# 会创建在脚本同目录下的 "faiss_index" 文件夹
DEFAULT_FAISS_DIR = ROOT / "faiss_index"


# ============================================================
# 辅助函数定义
# ============================================================

def _print_block(title: str, body: str) -> None:
    """
    打印格式化的标题块，用于在控制台输出时区分不同的章节

    参数:
        title: 标题文本
        body: 正文文本（支持多行，会自动处理缩进）

    效果:
        打印类似以下格式的输出：
        ======================================
        标题内容
        ======================================
        正文内容...
    """
    # 生成长度为70的分割线
    line = "=" * 70
    # 打印标题和分割线
    print(f"\n{line}\n{title}\n{line}")
    # textwrap.dedent() 移除 body 中的公共缩进，保持格式整洁
    print(textwrap.dedent(body).strip() + "\n")


def _optional_import(module: str, pip_hint: str) -> Optional[Any]:
    """
    尝试动态导入模块，如果失败则返回 None 并打印友好的错误提示

    这是一个软依赖处理机制：某些功能需要特定的可选模块（如 FAISS），
    如果这些模块未安装，脚本不会崩溃，而是优雅地跳过相关功能。

    参数:
        module: 要导入的模块名称（如 "faiss"、"langchain_community"）
        pip_hint: 安装提示，告诉用户如何安装缺失的模块

    返回:
        成功导入返回模块对象，失败返回 None

    工作流程:
        1. 尝试使用 __import__() 动态导入模块
        2. 如果失败，调用 _print_block() 显示友好错误信息
        3. 返回 None 表示模块不可用
    """
    try:
        # __import__() 是 Python 的动态导入函数
        # fromlist=["*"] 确保可以访问模块的顶层属性
        return __import__(module, fromlist=["*"])
    except Exception as e:
        # 捕获所有异常，避免脚本崩溃
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
    检查环境变量（或全局变量）是否已设置

    参数:
        name: 环境变量名称（如 "DASHSCOPE_API_KEY"）

    返回:
        已设置返回 True，否则返回 False 并打印提示

    注意:
        这里直接检查全局变量 DASHSCOPE_API_KEY
        实际项目中可以使用 os.environ.get(name) 检查环境变量
    """
    # 检查全局 API 密钥是否已设置
    if DASHSCOPE_API_KEY:
        return True
    # 未设置时打印友好提示
    _print_block(
        f"缺少环境变量：{name}",
        f"""
        当前未设置 `{name}`，对应示例可能无法运行。
        """,
    )
    return False


def _sample_documents() -> list[dict[str, Any]]:
    """
    返回内置示例文档列表，统一用于向量库示例

    当外部资源文件（如 resource/knowledge.txt）不存在时，使用这些内置的示例文档。
    这些文档都是关于 AI/LangChain 领域的简短描述，适合用于向量相似度搜索演示。

    返回:
        包含字典的列表，每个字典有两个键：
        - "page_content": 文档的文本内容
        - "metadata": 元数据字典（包含 "source" 字段标识来源）

    示例文档内容:
        1. Pika 公司是一家专注于 AI 视频生成的公司。
        2. Pika 由斯坦福大学的研究团队创立，致力于降低视频制作门槛。
        3. Chroma 是轻量易用的开源向量数据库，适合原型验证。
        4. 余弦相似度用于衡量向量夹角的相似程度。
        5. LangChain 是一个用于构建 LLM 应用的框架。
    """
    # 定义示例文档的文本内容
    texts = [
        "Pika 公司是一家专注于 AI 视频生成的公司。",
        "Pika 由斯坦福大学的研究团队创立，致力于降低视频制作门槛。",
        "Chroma 是轻量易用的开源向量数据库，适合原型验证。",
        "余弦相似度用于衡量向量夹角的相似程度。",
        "LangChain 是一个用于构建 LLM 应用的框架。",
    ]
    # 构建符合 LangChain Document 格式的字典列表
    # 每个文档包含 page_content（文本内容）和 metadata（元数据）
    return [{"page_content": t, "metadata": {"source": "builtin"}} for t in texts]


def _docs_to_langchain_documents(raw_docs: list[dict[str, Any]]):
    """
    将原始文档字典列表转换为 LangChain Document 对象列表

    LangChain 的向量数据库操作需要使用 langchain_core.documents.Document 对象，
    这个函数将简单的字典格式转换为 LangChain 标准的 Document 格式。

    参数:
        raw_docs: 原始文档字典列表，每个字典应包含 "page_content" 和可选的 "metadata"

    返回:
        LangChain Document 对象列表，如果 langchain-core 未安装则返回 None

    转换过程:
        输入: [{"page_content": "文本", "metadata": {"key": "value"}}]
        输出: [Document(page_content="文本", metadata={"key": "value"})]
    """
    # 尝试导入 langchain_core.documents 模块
    langchain_core_documents = _optional_import(
        "langchain_core.documents",
        "pip install -U langchain-core",
    )
    if langchain_core_documents is None:
        return None

    # 获取 Document 类
    Document = getattr(langchain_core_documents, "Document")

    # 转换：遍历原始文档列表，转换为 Document 对象
    # 每个字典的 page_content 转为 Document 的 page_content
    # metadata 保持不变
    return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw_docs]


# ============================================================
# 主演示函数
# ============================================================

def demo_faiss(persist_dir: Path, reset: bool = False) -> None:
    """
    FAISS 向量数据库示例

    FAISS (Facebook AI Similarity Search) 是 Facebook 开源的向量相似度搜索库，
    能够在海量向量数据中快速找到最相似的向量。它支持多种索引算法，
    能够在内存中处理数十亿级别的向量检索。

    本函数演示以下功能：
    1. 文档加载与分词 - 从文件或内置示例加载文档
    2. Embedding 生成 - 使用阿里云百炼的 text-embedding-v4 模型
    3. 构建 FAISS 索引 - 将文档向量存储到 FAISS 索引中
    4. 相似度搜索 - 执行基本的相似度查询
    5. 检索器模式 - 将 FAISS 作为 LangChain 检索器使用
    6. 带分数搜索 - 返回相似度分数
    7. 持久化 - 保存和加载索引

    参数:
        persist_dir: FAISS 索引持久化保存的目录路径
        reset: 是否在运行前清理持久化目录

    依赖库:
        - faiss-cpu: FAISS 核心库（CPU 版本）
        - langchain-community: LangChain 社区集成
        - langchain-openai: OpenAI 的 LangChain 集成（这里用于阿里云百炼）
        - langchain-text-splitters: 文本分词工具
        - langchain-core: LangChain 核心模块
    """
    # 打印标题块，说明本示例的内容
    _print_block(
        "FAISS 向量数据库",
        f"""
        持久化目录：{persist_dir}
        """,
    )

    # ============================================================
    # 清理持久化目录（如需要）
    # ============================================================
    # 如果 reset=True 且目录已存在，则删除整个目录树
    # 用于重新构建索引的情况
    if reset and persist_dir.exists():
        # shutil.rmtree 递归删除目录及其所有内容
        shutil.rmtree(persist_dir)
        print(f"已清理持久化目录：{persist_dir}")

    # ============================================================
    # 检查环境变量
    # ============================================================
    # FAISS 需要 API Key 来生成文本嵌入（Embeddings）
    # 这里使用阿里云百炼的 API
    if not _require_env("DASHSCOPE_API_KEY"):
        return

    # ============================================================
    # 导入必要的依赖库
    # ============================================================
    # langchain_community 提供了 LangChain 与各种向量数据库的集成
    langchain_community = _optional_import("langchain_community", "pip install -U langchain-community")
    if langchain_community is None:
        return

    # FAISS 核心库
    # 注意：faiss-cpu 是 CPU 版本，还有 faiss-gpu（需要 CUDA）
    faiss_mod = _optional_import("faiss", "pip install -U faiss-cpu")
    if faiss_mod is None:
        return

    # LangChain OpenAI 集成（这里用于阿里云百炼 API 的封装）
    openai_mod = _optional_import("langchain_openai", "pip install -U langchain-openai")
    if openai_mod is None:
        return

    # ============================================================
    # 导入具体的类
    # ============================================================
    # 注意：使用 from ... import 语法直接导入，比 getattr 更简洁
    # 但放在函数内部可以在依赖缺失时提前返回，避免导入错误

    # TextLoader: 用于从文本文件加载文档
    # 参数: 文件路径, 编码格式
    from langchain_community.document_loaders import TextLoader

    # FAISS: LangChain 封装的 FAISS 向量数据库类
    # 提供了 from_documents, similarity_search, save_local, load_local 等方法
    from langchain_community.vectorstores import FAISS

    # OpenAIEmbeddings: 文本嵌入模型
    # 这里使用阿里云百炼的 text-embedding-v4 模型，输出 1536 维向量
    from langchain_openai import OpenAIEmbeddings

    # CharacterTextSplitter: 按字符数分割文本的分割器
    # chunk_size: 每个块的最大字符数
    # chunk_overlap: 块之间的重叠字符数
    from langchain_text_splitters import CharacterTextSplitter

    # ============================================================
    # 加载文档
    # ============================================================
    # 尝试从资源文件加载，否则使用内置示例文档
    # 资源文件路径: 与脚本同目录下的 resource/knowledge.txt
    resource_file = ROOT / "resource" / "knowledge.txt"

    if resource_file.exists():
        # 使用 TextLoader 读取文本文件
        loader = TextLoader(str(resource_file), encoding="utf-8")
        documents = loader.load()

        # 使用 CharacterTextSplitter 将长文档分割成较小的块
        # chunk_size=1500: 每个块最多 1500 个字符
        # chunk_overlap=0: 块之间不重叠
        splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        docs = splitter.split_documents(documents)
    else:
        # 资源文件不存在，使用内置示例文档
        print(f"未找到 {resource_file}，使用内置示例文档。")
        raw_docs = _sample_documents()
        docs = _docs_to_langchain_documents(raw_docs)
        if docs is None:
            return
        # 内置文档已经很短，不需要再分割

    # ============================================================
    # 创建嵌入模型
    # ============================================================
    # OpenAIEmbeddings 会调用 API 将文本转换为向量
    # 这里使用阿里云百炼的 API，model 指定为 text-embedding-v4
    # 注意：虽然导入的是 OpenAIEmbeddings，但通过 base_url 可以兼容其他 API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",  # 使用的嵌入模型
        api_key=DASHSCOPE_API_KEY,  # 阿里云百炼 API 密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # API 端点
        check_embedding_ctx_length=False,  # 跳过上下文长度检查（百炼模型可能不同）
    )

    # ============================================================
    # 构建 FAISS 索引
    # ============================================================
    # FAISS.from_documents() 接受文档列表和嵌入模型
    # 内部会自动：
    #   1. 对每个文档调用嵌入模型生成向量
    #   2. 构建 FAISS 索引结构
    # 返回的 db 对象即为 FAISS 向量数据库实例
    db = FAISS.from_documents(docs, embeddings)
    print("FAISS 索引构建完成。\n")

    # ============================================================
    # 定义查询语句
    # ============================================================
    # 这个问题将用于后续的相似度搜索
    query = "Pika 公司是做什么的？"

    # ============================================================
    # 示例 1: 相似性搜索 (Similarity Search)
    # ============================================================
    # 最基本的搜索方式：找到与查询最相似的文档
    # 返回的文档按相似度从高到低排序
    print(f"--- 相似性搜索 ---")
    hits = db.similarity_search(query)
    print(f"查询：{query}")
    for i, d in enumerate(hits, start=1):
        print(f"  [Top {i}] {d.page_content}")

    # ============================================================
    # 示例 2: 作为检索器使用 (Retriever)
    # ============================================================
    # FAISS 实现了 LangChain 的 Retriever 接口
    # 可以直接转换为检索器，用于 LangChain 的链式调用（如 RAG）
    # 检索器默认返回 4 个最相似的文档
    print(f"\n--- 作为检索器 ---")
    retriever = db.as_retriever()
    # 使用 invoke 方法调用检索器
    retriever_docs = retriever.invoke(query)
    print(f"查询：{query}")
    for i, d in enumerate(retriever_docs, start=1):
        print(f"  [Top {i}] {d.page_content}")

    # ============================================================
    # 示例 3: 带分数的相似性搜索
    # ============================================================
    # similarity_search_with_score 返回文档及其相似度分数
    # 分数通常是距离度量（越小越相似）或相似度度量（越大越相似）
    # 取决于索引类型（FAISS 默认使用 L2 距离）
    print(f"\n--- 带分数的相似性搜索 ---")
    docs_and_scores = db.similarity_search_with_score(query)
    print(f"查询：{query}")
    for d, score in docs_and_scores:
        # 分数是 L2 距离，所以越小越相似
        print(f"  [分数: {score:.4f}] {d.page_content}")

    # ============================================================
    # 示例 4: 持久化保存
    # ============================================================
    # 将 FAISS 索引保存到指定目录
    # 保存的内容包括：
    #   - 索引文件 (.faiss)：存储向量数据的索引结构
    #   - 文档文件 (.pkl)：存储原始文档内容
    print(f"\n--- 持久化保存 ---")
    # 确保目录存在（parents=True 创建父目录，exist_ok=True 目录已存在不报错）
    persist_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(persist_dir))
    print(f"索引已保存到：{persist_dir}")

    # ============================================================
    # 示例 5: 从磁盘加载
    # ============================================================
    # 从之前保存的目录加载 FAISS 索引
    # 参数说明：
    #   - persist_dir: 保存索引的目录路径
    #   - embeddings: 嵌入模型（加载时需要用于向量计算）
    #   - allow_dangerous_deserialization: 允许反序列化（本地文件可以放心使用）
    print(f"\n--- 从磁盘加载 ---")
    new_db = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    # 验证加载后的索引可以正常搜索
    loaded_hits = new_db.similarity_search(query)
    print(f"查询：{query}")
    for i, d in enumerate(loaded_hits, start=1):
        print(f"  [Top {i}] {d.page_content}")

    # 打印完成提示
    print("\nFAISS 示例演示完成！")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="LangChain FAISS 向量数据库示例")

    # 添加 --persist-dir 参数：指定 FAISS 索引持久化目录
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(DEFAULT_FAISS_DIR),
        help="FAISS 持久化目录",
    )

    # 添加 --reset 参数：运行前清理持久化目录
    # action="store_true" 表示这是一个布尔标志，不带值
    parser.add_argument(
        "--reset",
        action="store_true",
        help="运行前清理 persist-dir",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    # 将字符串路径转换为 Path 对象
    demo_faiss(Path(args.persist_dir), reset=args.reset)
