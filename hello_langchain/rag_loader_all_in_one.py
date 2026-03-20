"""
LangChain 自定义 RAG 加载器示例
包含：自定义文档加载器、BaseBlobParser、Blob API、通用加载器

==========================================
RAG 文档加载器核心概念
==========================================

【什么是文档加载器？】
在 RAG（检索增强生成）系统中，文档加载器负责将各种来源的文档加载为标准格式。
支持的文件类型：
- 纯文本文件 (.txt)
- JSON 文件 (.json)
- Markdown 文件 (.md)
- PDF 文件 (.pdf)
- Word 文档 (.docx)
- HTML 文件 (.html)
- CSV 文件 (.csv)
- 等等...

【LangChain 文档加载器架构】

┌─────────────────────────────────────────────────────────────┐
│                    数据源                                   │
│   (文件、数据库、API、网页、S3 等)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  BaseBlobLoader / BaseLoader                               │
│  负责：读取原始数据，转换为 Blob 或直接输出 Document         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  BaseBlobParser                                             │
│  负责：解析 Blob（原始二进制数据），生成 Document            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Document                                                   │
│  page_content: 文档内容（字符串）                            │
│  metadata: 元数据（字典，如 source、line_number 等）         │
└─────────────────────────────────────────────────────────────┘

【核心组件详解】

1. Document（文档）
   - LangChain 的标准文档格式
   - page_content: 文档的实际文本内容
   - metadata: 元数据字典，包含来源、行号、创建时间等信息

2. Blob（二进制大对象）
   - 表示原始二进制数据的容器
   - 可以来自文件、内存或网络
   - 提供统一的 API 访问数据
   - 支持编码转换（字节 ↔ 字符串）

3. BaseLoader（基础加载器）
   - 最简单的加载器接口
   - 直接从数据源加载为 Document
   - 适合简单的数据源

4. BaseBlobParser（Blob 解析器）
   - 解析 Blob 生成 Document
   - 将二进制数据转换为文本
   - 支持分块、格式化等处理

5. BaseBlobLoader（Blob 加载器）
   - 读取数据源，生成 Blob
   - 负责数据的读取和传输

6. DirectoryLoader（目录加载器）
   - 批量加载目录中的文件
   - 支持 glob 模式匹配
   - 可配合自定义加载器使用

【惰性加载（Lazy Loading）】
- 使用生成器（yield）按需加载数据
- 优点：内存效率高，适合处理大文件
- 与 load() 的区别：load() 一次性加载所有数据

【本文件涵盖的内容】
1. CustomDocumentLoader: 继承 BaseLoader 的自定义加载器
2. LineParser: 继承 BaseBlobParser 的行解析器
3. Blob API: Blob 的创建和基本操作
4. DirectoryLoader: 使用 LangChain 提供的目录加载器
5. CustomFileLoader: 结合 BlobLoader 和 Parser 的加载器
6. JSONLoader: 处理 JSON 文件的自定义加载器

依赖安装：
pip install aiofiles  # 用于异步文件操作
"""

import os
import json
import shutil
from typing import AsyncIterator, Iterator, Optional, Union

# 获取当前文件所在目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 测试文件目录 - 保存在项目目录的 doc/test_files 下
TEST_FILES_DIR = os.path.join(_CURRENT_DIR, "doc", "test_files")

# LangChain 核心组件
from langchain_core.document_loaders import BaseLoader, BaseBlobParser, Blob
from langchain_core.documents import Document

# LangChain 社区提供的加载器
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader

# 自定义支持 UTF-8 编码的 TextLoader
from langchain_community.document_loaders import TextLoader as BaseTextLoader


def _ensure_test_dir():
    """
    确保测试文件目录存在
    
    目录结构：
    hello_langchain/
    └── doc/
        └── test_files/  <- 测试文件放在这里
            ├── demo_custom_loader/
            │   └── meow.txt
            ├── demo_blob_parser/
            │   └── pets.txt
            └── ...
    """
    if not os.path.exists(TEST_FILES_DIR):
        os.makedirs(TEST_FILES_DIR, exist_ok=True)
        print(f"创建测试文件目录: {TEST_FILES_DIR}")


class UTF8TextLoader(BaseTextLoader):
    """
    支持 UTF-8 编码的 TextLoader
    
    【为什么需要自定义？】
    langchain 的 BaseTextLoader 默认使用系统编码（Windows 上可能是 GBK）
    对于 UTF-8 编码的文件，需要显式指定编码。
    
    继承自 BaseTextLoader，只覆盖 lazy_load 方法中的编码设置。
    """

    def lazy_load(self):
        """惰性加载文档（使用 UTF-8 编码）"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        yield Document(
            page_content=content,
            metadata={"source": self.file_path}
        )


# ============================================================
# 组件概述
# ============================================================
# 以下表格展示了 LangChain 文档加载器体系中的核心组件：

"""
| 组件           | 描述                                              |
|----------------|---------------------------------------------------|
| Document       | 包含 page_content 和 metadata 的文档对象          |
| BaseLoader     | 基础加载器接口，从数据源直接加载为 Document       |
| BaseBlobParser | Blob 解析器，将 Blob 转换为 Document              |
| Blob           | 二进制大对象，表示文件或内存中的原始数据           |
| BaseBlobLoader | Blob 加载器，读取数据源生成 Blob                  |
| DirectoryLoader| 目录加载器，批量加载目录中的文件                  |
"""


# ============================================================
# 一、自定义文档加载器（继承 BaseLoader）
# ============================================================
#
# 【BaseLoader 适用场景】
# - 数据源是单一文件或简单的 API
# - 不需要复杂的二进制数据处理
# - 直接读取后立即转换为 Document
#
# 【必须实现的方法】
# - lazy_load(): 返回 Iterator[Document] 的生成器
# - alazy_load(): 可选，异步版本的惰性加载

class CustomDocumentLoader(BaseLoader):
    """
    自定义文档加载器 - 从文件逐行加载文档
    
    【设计目的】
    演示如何继承 BaseLoader 创建自定义加载器。
    这个加载器逐行读取文件，每行作为一个 Document。
    
    【与 BaseBlobParser 的区别】
    - BaseLoader: 直接从数据源读取并生成 Document
    - BaseBlobParser: 解析 Blob（需要先有 Blob）
    
    【为什么需要 lazy_load？】
    - 惰性加载：按需读取，不一次性加载整个文件
    - 适合处理大文件，不会占用过多内存
    - 返回生成器（Generator），节省资源
    
    Attributes:
        file_path: 要加载的文件路径
    """

    def __init__(self, file_path: str) -> None:
        """
        初始化加载器
        
        Args:
            file_path: 要加载的文件路径（支持绝对路径和相对路径）
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载文档（同步版本）
        
        【返回类型】
        Iterator[Document]：文档生成器
        
        【工作原理】
        1. 打开文件
        2. 逐行读取
        3. 每行创建一个 Document
        4. yield 返回（不一次性返回所有）
        
        【yield 的作用】
        - 生成器函数，不是普通函数
        - 每次调用返回一个值后暂停执行
        - 下次调用时从暂停处继续
        - 节省内存，适合大文件处理
        
        Yields:
            Document: 每次返回一个文档，包含行内容和行号
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                # 创建 Document 对象
                # page_content: 行内容
                # metadata: 包含行号和来源文件
                yield Document(
                    page_content=line,
                    metadata={
                        "line_number": line_number,  # 行号（从0开始）
                        "source": self.file_path,     # 来源文件路径
                    },
                )
                line_number += 1

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        惰性加载文档（异步版本）
        
        【为什么需要异步版本？】
        - 提高 I/O 效率：在等待文件读取时可以处理其他任务
        - 支持高并发：可以同时处理多个文件
        - 非阻塞：不会卡住主线程
        
        【注意】
        - 需要 aiofiles 库支持异步文件操作
        - pip install aiofiles
        - 如果未安装，会抛出 ImportError
        
        Yields:
            Document: 每次返回一个文档
        """
        try:
            import aiofiles  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "aiofiles 未安装。请运行: pip install aiofiles"
            )

        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            async for line in f:
                yield Document(
                    page_content=line,
                    metadata={
                        "line_number": line_number,
                        "source": self.file_path,
                    },
                )
                line_number += 1


def demo_custom_loader():
    """
    自定义加载器演示
    
    演示如何使用 CustomDocumentLoader 加载文件。
    
    【测试流程】
    1. 创建测试文件到 doc/test_files/demo_custom_loader/
    2. 实例化加载器
    3. 使用惰性加载（生成器）
    4. 使用全部加载（load 方法）
    5. （可选）清理测试文件
    """
    print("\n" + "=" * 50)
    print("1. 自定义 BaseLoader")
    print("=" * 50)

    # 确保测试目录存在
    _ensure_test_dir()

    # =========================================================================
    # 【创建测试文件】
    # =========================================================================
    # 目录结构: doc/test_files/demo_custom_loader/meow.txt
    demo_dir = os.path.join(TEST_FILES_DIR, "demo_custom_loader")
    os.makedirs(demo_dir, exist_ok=True)
    
    test_file = os.path.join(demo_dir, "meow.txt")
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("喵喵喵\n")   # 第一行
        f.write("喵呜\n")     # 第二行
        f.write("喵~")        # 第三行（无换行符）

    print(f"测试文件: {test_file}")
    print(f"文件内容:")
    with open(test_file, "r", encoding="utf-8") as f:
        print(f.read())

    # =========================================================================
    # 【使用加载器】
    # =========================================================================
    loader = CustomDocumentLoader(test_file)

    # -------------------------------------------------------------------------
    # 【惰性加载】
    # -------------------------------------------------------------------------
    print("\n惰性加载（lazy_load）:")
    
    # lazy_load 返回生成器，需要遍历获取数据
    # 每次只读取一行，内存效率高
    for doc in loader.lazy_load():
        # repr() 显示原始字符串（包括换行符 \n）
        print(f"  内容: {repr(doc.page_content)}, 元数据: {doc.metadata}")

    # -------------------------------------------------------------------------
    # 【全部加载】
    # -------------------------------------------------------------------------
    print("\n全部加载（load）:")
    
    # load() 是 BaseLoader 提供的方法
    # 会一次性加载所有文档到列表
    # 适合小文件，大文件可能导致内存问题
    docs = loader.load()
    
    for doc in docs:
        print(f"  {doc.page_content.strip()}")  # strip() 去除首尾空白

    # =========================================================================
    # 【提示】
    # =========================================================================
    # 测试文件已保存在项目中，不会自动删除
    # 路径: doc/test_files/demo_custom_loader/meow.txt


# ============================================================
# 二、BaseBlobParser（Blob 解析器）
# ============================================================
#
# 【BaseBlobParser 适用场景】
# - 需要处理二进制数据
# - 同一个解析逻辑可用于多个数据源
# - 需要灵活的数据转换处理
#
# 【必须实现的方法】
# - lazy_parse(blob: Blob): 返回 Iterator[Document]

class LineParser(BaseBlobParser):
    """
    行解析器 - 将 Blob 按行分割为多个 Document
    
    【设计目的】
    演示如何继承 BaseBlobParser 创建自定义解析器。
    这个解析器将 Blob 的内容按行分割，每行生成一个 Document。
    
    【与 BaseLoader 的区别】
    - BaseLoader: 自己读取数据源
    - BaseBlobParser: 接收 Blob 作为输入，由外部提供数据源
    
    【as_bytes_io() 方法】
    - 将 Blob 转换为字节流
    - 返回 BytesIO 对象，可以像文件一样读取
    - 支持 .read()、.seek() 等文件操作
    
    Attributes:
        encoding: 文本编码，默认 utf-8
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        """
        初始化解析器
        
        Args:
            encoding: 文本编码，默认 utf-8
        """
        self.encoding = encoding

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        解析 Blob，每行生成一个 Document
        
        【Blob 参数】
        - 包含原始二进制数据
        - 可以来自文件、内存或网络
        
        【处理流程】
        1. 使用 blob.as_bytes_io() 获取字节流
        2. 逐行读取字节流
        3. 解码为字符串
        4. 创建 Document
        
        【注意】
        这里直接遍历字节流，未做解码处理。
        实际使用中应根据编码正确解码。
        
        Args:
            blob: 要解析的 Blob 对象
        
        Yields:
            Document: 每次返回一个文档
        """
        line_number = 0
        
        # as_bytes_io() 返回 BytesIO 对象
        # 可以像文件一样使用 with 语句
        with blob.as_bytes_io() as f:
            for line in f:
                line_number += 1
                yield Document(
                    page_content=line,  # 字节内容，实际应用需解码
                    metadata={
                        "line_number": line_number,
                        "source": blob.source,  # 来自 Blob 的 source 属性
                    },
                )


def demo_blob_parser():
    """
    BaseBlobParser 演示
    
    演示如何使用 LineParser 解析 Blob。
    
    【测试流程】
    1. 创建测试文件
    2. 创建 Blob 对象
    3. 使用解析器处理 Blob
    """
    print("\n" + "=" * 50)
    print("2. BaseBlobParser")
    print("=" * 50)

    # 确保测试目录存在
    _ensure_test_dir()

    # =========================================================================
    # 【创建测试文件】
    # =========================================================================
    demo_dir = os.path.join(TEST_FILES_DIR, "demo_blob_parser")
    os.makedirs(demo_dir, exist_ok=True)
    
    test_file = os.path.join(demo_dir, "pets.txt")
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("狗是忠诚的伙伴\n")
        f.write("猫是独立的伴侣\n")
        f.write("兔子爱吃胡萝卜\n")

    print(f"测试文件: {test_file}")
    print(f"文件内容:")
    with open(test_file, "r", encoding="utf-8") as f:
        print(f.read())

    # =========================================================================
    # 【创建 Blob】
    # =========================================================================
    #
    # Blob.from_path() 创建一个指向文件的 Blob
    # Blob 包含文件的路径和元数据，不直接加载内容
    
    blob = Blob.from_path(test_file)
    print(f"\nBlob 源: {blob.source}")

    # =========================================================================
    # 【解析 Blob】
    # =========================================================================
    parser = LineParser()
    
    # lazy_parse 返回生成器
    docs = list(parser.lazy_parse(blob))

    print("解析结果:")
    for doc in docs:
        # page_content 是字节，需要解码显示
        content = doc.page_content
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        print(f"  行{doc.metadata['line_number']}: {content.strip()}")


# ============================================================
# 三、Blob API
# ============================================================
#
# 【Blob 简介】
# Blob = Binary Large Object（二进制大对象）
# 是 LangChain 中表示原始二进制数据的类
#
# 【Blob 的来源】
# 1. Blob.from_path(path): 从文件创建
# 2. Blob.from_data(data): 从内存数据创建
# 3. Blob.from_string(text): 从字符串创建

def demo_blob_api():
    """
    Blob API 演示
    
    演示 Blob 的创建方式和常用 API。
    
    【创建 Blob 的方式】
    1. from_path(): 从文件路径创建
    2. from_data(): 从字节数据创建
    3. from_string(): 从字符串创建
    """
    print("\n" + "=" * 50)
    print("3. Blob API")
    print("=" * 50)

    # 确保测试目录存在
    _ensure_test_dir()

    # =========================================================================
    # 【创建测试文件】
    # =========================================================================
    demo_dir = os.path.join(TEST_FILES_DIR, "demo_blob_api")
    os.makedirs(demo_dir, exist_ok=True)
    
    test_file = os.path.join(demo_dir, "hello.txt")
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, World!")

    print(f"测试文件: {test_file}")

    # =========================================================================
    # 【创建 Blob】
    # =========================================================================
    #
    # Blob.from_path() 从文件创建 Blob
    # metadata 参数可以传入自定义元数据
    
    blob = Blob.from_path(
        path=test_file,
        metadata={"foo": "bar"}  # 自定义元数据
    )

    # =========================================================================
    # 【访问 Blob 属性】
    # =========================================================================
    
    print(f"\n数据源: {blob.source}")    # 文件路径
    print(f"元数据: {blob.metadata}")  # 自定义 + 继承的元数据
    print(f"编码: {blob.encoding}")    # 文本编码

    # =========================================================================
    # 【Blob 数据转换方法】
    # =========================================================================
    
    # as_bytes(): 返回原始字节数据
    # 类型: bytes
    print(f"\n字节数据: {blob.as_bytes()}")

    # as_string(): 返回解码后的字符串
    # 自动使用 blob.encoding 解码
    print(f"字符串: {blob.as_string()}")

    # as_bytes_io(): 返回 BytesIO 对象
    # 可以像文件一样读取，支持 .read()、.seek() 等
    print(f"BytesIO 类型: {type(blob.as_bytes_io())}")

    # =========================================================================
    # 【创建带数据的 Blob】
    # =========================================================================
    
    # from_data(): 从字节数据创建
    blob_from_bytes = Blob.from_data(data=b"Binary data example")
    print(f"\n从字节创建的 Blob: {blob_from_bytes.as_string()}")

    # from_data() 同样可以从字符串创建（会自动编码为字节）
    blob_from_string = Blob.from_data(data="Text content")
    print(f"从字符串创建的 Blob: {blob_from_string.as_string()}")


# ============================================================
# 四、目录加载器（DirectoryLoader）
# ============================================================
#
# 【DirectoryLoader 简介】
# DirectoryLoader 是 langchain-community 提供的目录批量加载器。
# 可以递归扫描目录，加载所有匹配的文件。
#
# 【DirectoryLoader 参数】
# - path: 要加载的目录路径
# - glob: 文件匹配模式
# - loader_cls: 使用的加载器类（如 TextLoader）
# - show_progress: 是否显示进度条
# - recursive: 是否递归扫描子目录


def demo_directory_loader():
    """
    DirectoryLoader 演示
    
    演示如何使用 DirectoryLoader 从目录加载多个文件。
    
    【DirectoryLoader 特点】
    - 使用 loader_cls 指定加载器类
    - 支持 glob 模式匹配
    - 可选择是否递归扫描子目录
    """
    print("\n" + "=" * 50)
    print("4. DirectoryLoader")
    print("=" * 50)

    # 确保测试目录存在
    _ensure_test_dir()

    # =========================================================================
    # 【创建测试目录和文件】
    # =========================================================================
    demo_dir = os.path.join(TEST_FILES_DIR, "demo_directory_loader")
    os.makedirs(demo_dir, exist_ok=True)
    
    test_file1 = os.path.join(demo_dir, "doc1.txt")
    test_file2 = os.path.join(demo_dir, "doc2.txt")

    # 写入第一个文件
    with open(test_file1, "w", encoding="utf-8") as f:
        f.write("这是第一个文档\n")
        f.write("包含两行内容")

    # 写入第二个文件
    with open(test_file2, "w", encoding="utf-8") as f:
        f.write("这是第二个文档\n")
        f.write("也包含两行内容")

    print(f"测试目录: {demo_dir}")
    print(f"文件列表: {os.listdir(demo_dir)}")

    # =========================================================================
    # 【创建 DirectoryLoader】
    # =========================================================================
    #
    # DirectoryLoader 使用 loader_cls 参数指定加载器类
    # 使用自定义的 UTF8TextLoader 确保正确读取 UTF-8 文件
    
    loader = DirectoryLoader(
        path=demo_dir,
        glob="*.txt",
        loader_cls=UTF8TextLoader,  # 使用支持 UTF-8 的加载器
        show_progress=False         # 不显示进度条
    )

    # =========================================================================
    # 【加载文档】
    # =========================================================================
    print("\n加载结果:")
    
    for doc in loader.load():  # DirectoryLoader 使用 load() 方法
        source_path = doc.metadata.get("source", "未知")
        filename = os.path.basename(source_path)
        
        print(f"  来源: {filename}")
        print(f"  内容: {doc.page_content.strip()}")
        print()

    # =========================================================================
    # 【使用 lazy_load 惰性加载】
    # =========================================================================
    print("惰性加载结果:")
    
    for doc in loader.lazy_load():
        source_path = doc.metadata.get("source", "未知")
        filename = os.path.basename(source_path)
        
        print(f"  来源: {filename}")
        print(f"  内容: {doc.page_content.strip()[:30]}...")
        print()


# ============================================================
# 五、自定义通用加载器（结合 BlobLoader 和 Parser）
# ============================================================
#
# 【设计模式】
# 这种方式组合了 BlobLoader 和 Parser：
# - BlobLoader: 负责读取文件，生成 Blob
# - Parser: 负责解析 Blob，生成 Document
#
# 【FileSystemBlobLoader】
# LangChain 提供的文件系统 Blob 加载器。
# 可以指定目录和文件模式，支持递归扫描。


class CustomFileLoader(BaseLoader):
    """
    自定义文件加载器 - 结合 BlobLoader 和 Parser
    
    【设计目的】
    演示如何组合使用 BlobLoader 和 Parser。
    这种方式比直接继承 BaseLoader 更灵活。
    
    【工作流程】
    1. FileSystemBlobLoader 遍历目录，找到所有匹配的文件
    2. 每个文件生成一个 Blob
    3. LineParser 解析每个 Blob，生成 Document
    
    【适用场景】
    - 需要处理多种文件类型（使用不同的 Parser）
    - 需要在加载过程中进行复杂处理
    - 需要复用已有的 BlobLoader 或 Parser
    
    Attributes:
        path: 要加载的目录路径
    """

    def __init__(self, path: str) -> None:
        """
        初始化加载器
        
        Args:
            path: 要加载的目录路径
        """
        self.path = path

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载目录中的所有文件
        
        【处理流程】
        1. 创建 FileSystemBlobLoader，指定目录
        2. 创建 LineParser 解析器
        3. 遍历 BlobLoader 产生的 Blob
        4. 每个 Blob 用 Parser 解析为 Document
        5. yield 返回所有 Document
        
        Yields:
            Document: 目录中所有文件的文档
        """
        # 创建文件系统 Blob 加载器
        # 它会遍历目录，返回所有 Blob
        blob_loader = FileSystemBlobLoader(path=self.path)
        
        # 创建解析器
        parser = LineParser()
        
        # 遍历所有 Blob
        for blob in blob_loader.yield_blobs():
            # yield from 委托给 parser 的生成器
            yield from parser.lazy_parse(blob)


def demo_custom_file_loader():
    """
    自定义通用加载器演示
    
    演示如何使用 CustomFileLoader 加载目录中的所有文件。
    """
    print("\n" + "=" * 50)
    print("5. 自定义通用加载器")
    print("=" * 50)

    # 确保测试目录存在
    _ensure_test_dir()

    # =========================================================================
    # 【创建测试目录和文件】
    # =========================================================================
    demo_dir = os.path.join(TEST_FILES_DIR, "demo_custom_file_loader")
    os.makedirs(demo_dir, exist_ok=True)
    
    test_file = os.path.join(demo_dir, "data.txt")

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("第一行数据\n")
        f.write("第二行数据\n")
        f.write("第三行数据")

    print(f"测试目录: {demo_dir}")

    # =========================================================================
    # 【使用自定义加载器】
    # =========================================================================
    loader = CustomFileLoader(demo_dir)

    print("\n加载结果:")
    for doc in loader.lazy_load():
        content = doc.page_content
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        print(f"  行号 {doc.metadata['line_number']}: {content.strip()}")


# ============================================================
# 六、完整的自定义加载器示例（JSON 文件）
# ============================================================
#
# 【JSON 加载器的重要性】
# JSON 是现代 Web API 和数据存储的标准格式。
# 很多 RAG 应用需要从 JSON 文件加载数据。
#
# 【jq_schema 参数】
# 类似于 jq 工具的查询语法，用于从复杂的 JSON 中提取数据。
# - ".": 整个 JSON 对象
# - "data": 提取 JSON 中 key 为 "data" 的部分
# - "data.items": 提取嵌套的 "data.items"


class JSONLoader(BaseLoader):
    """
    自定义 JSON 加载器
    
    【设计目的】
    演示如何为特定文件格式创建加载器。
    这个加载器处理 JSON 文件，支持简单的路径提取。
    
    【支持的 JSON 格式】
    1. JSON 数组: [{"key": "value1"}, {"key": "value2"}]
       - 每个元素作为一个 Document
    2. JSON 对象: {"items": [...], "meta": {...}}
       - 可以通过 jq_schema 提取特定字段
    3. JSON 字符串: "简单字符串"
       - 整个字符串作为一个 Document
    
    【jq_schema 的使用】
    - jq_schema=".": 加载整个 JSON
    - jq_schema="data": 加载 JSON["data"] 字段
    - jq_schema="results": 加载 JSON["results"] 字段
    
    Attributes:
        file_path: JSON 文件路径
        jq_schema: JSON 路径表达式，用于提取数据
    """

    def __init__(
        self,
        file_path: str,
        jq_schema: str = "."
    ) -> None:
        """
        初始化 JSON 加载器
        
        Args:
            file_path: JSON 文件的路径
            jq_schema: JSON 路径表达式，默认为 "." 表示整个 JSON
        """
        self.file_path = file_path
        self.jq_schema = jq_schema

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载 JSON 文件
        
        【处理流程】
        1. 读取 JSON 文件
        2. 根据 jq_schema 提取数据
        3. 如果是数组，遍历每个元素
        4. 每个元素创建一个 Document
        5. page_content 存储 JSON 字符串
        6. metadata 存储索引和来源
        
        Yields:
            Document: 每个 JSON 元素对应的文档
        """
        # 读取并解析 JSON 文件
        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)

        # -------------------------------------------------------------------------
        # 【根据 jq_schema 提取数据】
        # -------------------------------------------------------------------------
        #
        # jq_schema=".": 加载整个 JSON
        #   - 如果是数组，直接使用
        #   - 如果是对象，包装成数组
        #
        # jq_schema="key": 加载 JSON[key]
        #   - 如果 key 不存在，返回空数组
        
        if self.jq_schema == ".":
            # "." 表示整个 JSON
            if isinstance(data, list):
                # JSON 是数组，每个元素作为一个文档
                items = data
            else:
                # JSON 是对象，包装成数组
                items = [data]
        elif isinstance(data, dict):
            # jq_schema 指定了字段名
            items = data.get(self.jq_schema, [])
        else:
            # JSON 不是对象，无法使用 jq_schema
            items = []

        # -------------------------------------------------------------------------
        # 【创建 Document】
        # -------------------------------------------------------------------------
        
        for idx, item in enumerate(items):
            # 将 Python 对象转换为 JSON 字符串
            # ensure_ascii=False: 不转义中文等非 ASCII 字符
            page_content = json.dumps(
                item,
                ensure_ascii=False,
                indent=2  # 格式化输出，便于阅读
            )
            
            yield Document(
                page_content=page_content,
                metadata={
                    "index": idx,              # 在数组中的索引
                    "source": self.file_path,   # 来源文件
                    "jq_schema": self.jq_schema # 使用的 schema
                },
            )


def demo_json_loader():
    """
    JSON 加载器演示
    
    演示如何使用 JSONLoader 加载 JSON 文件。
    
    【测试场景】
    1. 创建包含多个对象的 JSON 数组
    2. 加载并解析
    3. 展示 jq_schema 的使用
    """
    print("\n" + "=" * 50)
    print("6. JSON 加载器")
    print("=" * 50)

    # 确保测试目录存在
    _ensure_test_dir()

    # =========================================================================
    # 【创建测试 JSON 文件】
    # =========================================================================
    demo_dir = os.path.join(TEST_FILES_DIR, "demo_json_loader")
    os.makedirs(demo_dir, exist_ok=True)
    
    test_file = os.path.join(demo_dir, "data.json")
    
    # 模拟用户数据
    test_data = [
        {"name": "张三", "age": 30, "city": "北京"},
        {"name": "李四", "age": 25, "city": "上海"},
        {"name": "王五", "age": 35, "city": "深圳"},
    ]

    # 写入 JSON 文件
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"测试文件: {test_file}")

    # =========================================================================
    # 【使用加载器 - 默认 schema】
    # =========================================================================
    print("\n--- 使用默认 schema (\".\") ---")
    
    loader = JSONLoader(test_file)
    
    print("加载结果:")
    for doc in loader.lazy_load():
        print(f"  索引: {doc.metadata['index']}")
        print(f"  内容: {doc.page_content}")
        print()

    # =========================================================================
    # 【创建更复杂的 JSON 用于测试 jq_schema】
    # =========================================================================
    print("\n--- 测试 jq_schema ---")
    
    complex_file = os.path.join(demo_dir, "complex.json")
    complex_data = {
        "meta": {"version": "1.0", "total": 3},
        "users": [
            {"id": 1, "name": "用户A"},
            {"id": 2, "name": "用户B"},
            {"id": 3, "name": "用户C"},
        ]
    }
    
    with open(complex_file, "w", encoding="utf-8") as f:
        json.dump(complex_data, f, ensure_ascii=False, indent=2)

    # 使用 jq_schema="users" 只加载 users 字段
    loader_users = JSONLoader(complex_file, jq_schema="users")
    
    print("jq_schema=\"users\" 的结果:")
    for doc in loader_users.lazy_load():
        print(f"  {doc.page_content}")


# ============================================================
# 测试文件生成函数（用于外部测试）
# ============================================================
#
# 以下函数用于生成测试文件，方便在其他脚本中复用。

def create_test_text_file(
    content: str,
    filename: Optional[str] = None,
    subdir: Optional[str] = None
) -> str:
    """
    创建测试用的文本文件
    
    【存储位置】
    默认保存在 doc/test_files/ 目录下
    
    Args:
        content: 文件内容
        filename: 文件名（可选，默认自动生成）
        subdir: 子目录名（可选）
    
    Returns:
        str: 创建的文件路径
    """
    _ensure_test_dir()
    
    if subdir:
        target_dir = os.path.join(TEST_FILES_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = TEST_FILES_DIR
    
    if filename is None:
        filename = f"test_{hash(content) % 10000}.txt"
    
    filepath = os.path.join(target_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath


def create_test_json_file(
    data: Union[list, dict],
    filename: Optional[str] = None,
    subdir: Optional[str] = None
) -> str:
    """
    创建测试用的 JSON 文件
    
    【存储位置】
    默认保存在 doc/test_files/ 目录下
    
    Args:
        data: JSON 数据
        filename: 文件名（可选，默认自动生成）
        subdir: 子目录名（可选）
    
    Returns:
        str: 创建的文件路径
    """
    _ensure_test_dir()
    
    if subdir:
        target_dir = os.path.join(TEST_FILES_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = TEST_FILES_DIR
    
    if filename is None:
        filename = f"test_{hash(str(data)) % 10000}.json"
    
    filepath = os.path.join(target_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath


def cleanup_test_files(keep_structure: bool = True):
    """
    清理测试文件
    
    Args:
        keep_structure: 是否保留目录结构
            - True: 只删除测试文件，保留目录
            - False: 删除整个 test_files 目录
    """
    if os.path.exists(TEST_FILES_DIR):
        if keep_structure:
            # 只删除文件，保留目录
            for root, dirs, files in os.walk(TEST_FILES_DIR):
                for file in files:
                    filepath = os.path.join(root, file)
                    os.remove(filepath)
                    print(f"已删除: {filepath}")
        else:
            # 删除整个目录
            shutil.rmtree(TEST_FILES_DIR)
            print(f"已删除目录: {TEST_FILES_DIR}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    """
    程序入口点
    
    运行所有演示：
    1. demo_custom_loader: 自定义 BaseLoader
    2. demo_blob_parser: BaseBlobParser
    3. demo_blob_api: Blob API
    4. demo_directory_loader: DirectoryLoader
    5. demo_custom_file_loader: 自定义通用加载器
    6. demo_json_loader: JSON 加载器
    
    【测试文件存储位置】
    所有测试文件保存在: doc/test_files/
    """
    
    print("=" * 60)
    print("LangChain 自定义 RAG 加载器演示")
    print("=" * 60)
    print(f"\n测试文件保存位置: {TEST_FILES_DIR}")
    
    # 1. 自定义 BaseLoader - 最基础的加载器实现
    demo_custom_loader()
    
    # 2. BaseBlobParser - Blob 解析器
    demo_blob_parser()
    
    # 3. Blob API - Blob 的创建和操作
    demo_blob_api()
    
    # 4. DirectoryLoader - 目录加载器
    demo_directory_loader()
    
    # 5. 自定义通用加载器 - 结合 BlobLoader 和 Parser
    demo_custom_file_loader()
    
    # 6. JSON 加载器 - 处理 JSON 格式文件
    demo_json_loader()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print(f"\n测试文件保存在: {TEST_FILES_DIR}")
    print("如需清理，请调用 cleanup_test_files() 函数")
