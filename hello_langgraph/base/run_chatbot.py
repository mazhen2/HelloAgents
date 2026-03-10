# 添加项目根目录到 Python 路径
import sys
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
import os

os.environ["PYTHONPATH"] = str(project_root)

# 设置 UTF-8 输出
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def run_chatbot(graph_module: str, test_questions: list[str]):
    """运行聊天机器人测试的通用函数"""
    module = __import__(graph_module, fromlist=['graph'])
    graph = module.graph

    # 使用 thread_id 来标识对话会话，支持记忆功能
    config = {"configurable": {"thread_id": "test-thread"}}

    for question in test_questions:
        print(f"You: {question}")
        result = graph.invoke({"messages": [("human", question)]}, config)
        ai_message = result["messages"][-1].content
        print(f"AI: {ai_message}\n")
        print("-" * 30)


def run_base_chatbot():
    """测试基础聊天机器人"""
    test_questions = [
        "你好，请介绍一下自己"
    ]
    run_chatbot("hello_langgraph.base.chatbot", test_questions)


def run_add_tools_chatbot():
    """测试带工具的聊天机器人"""
    test_questions = [
        "现在几点了？",
        "北京天气怎么样？",
        "你好！"
    ]
    run_chatbot("hello_langgraph.base.add_tools_chatbot", test_questions)

def run_add_memory_chatbot():
    """测试带工具的聊天机器人"""
    test_questions = [
        "现在几点了？",
        "北京天气怎么样？",
        "我问你的第一个问题是什么？"
    ]
    run_chatbot("hello_langgraph.base.add_memory_chatbot", test_questions)


if __name__ == '__main__':
    # run_base_chatbot()
    # run_add_tools_chatbot()
    run_add_memory_chatbot()