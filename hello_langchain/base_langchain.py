import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 自动加载 .env（若存在）
load_dotenv()

# 从统一的 LLM_* 环境变量读取配置（与仓库 .env 约定一致）
model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
timeout = int(os.getenv("LLM_TIMEOUT", "60"))

if not api_key:
    raise ValueError("缺少API密钥：请在环境变量或 .env 中设置 LLM_API_KEY（或 OPENAI_API_KEY）。")
if not base_url:
    raise ValueError("缺少服务地址：请在环境变量或 .env 中设置 LLM_BASE_URL。")

# 初始化模型
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
    timeout=timeout,
)

# 创建提示模板
prompts = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术专家"),
    ("user", "{input}")
])

# 简单的链
print("简单的链")
chain = prompts | llm
print(chain.invoke({"input": "请写一个Python代码，实现一个简单的加法功能"}))

# 输出转换：只保留文本内容
print("输出转换：只保留文本内容")
output_parser = StrOutputParser()
chain_text = prompts | llm | output_parser
print(chain_text.invoke({"input": "请写一个Python代码，实现一个简单的加法功能"}))
