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
    """添加记忆"""
    test_questions = [
        "现在几点了？",
        "北京天气怎么样？",
        "我问你的第一个问题是什么？"
    ]
    run_chatbot("hello_langgraph.base.add_memory_chatbot", test_questions)


def run_add_human_in_the_loop_chatbot():
    """添加人工在环控制"""
    test_questions = [
        "北京天气怎么样？",
        "我问你的第一个问题是什么？"
    ]
    run_chatbot("hello_langgraph.base.add_human-in-the-loop_chatbot", test_questions)


def run_time_travel_chatbot():
    """测试时间旅行功能"""
    from hello_langgraph.base.time_travel_chatbot import graph

    # 创建配置字典，指定线程 ID
    config = {"configurable": {"thread_id": "time-travel-test"}}

    # -------------------------------------------------------------------------
    # 第一次调用：用户询问关于 LangGraph 的信息
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("第一次调用：用户询问关于 LangGraph 的信息")
    print("=" * 80)

    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "我正在学习 LangGraph。"
                        "你能帮我搜索一下相关信息吗？"
                    ),
                },
            ],
        },
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # -------------------------------------------------------------------------
    # 第二次调用：用户回应 AI 的回复
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("第二次调用：用户回应 AI 的回复")
    print("=" * 80)

    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "很有帮助！也许我会用它"
                        "构建一个自主代理！"
                    ),
                },
            ],
        },
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # -------------------------------------------------------------------------
    # 获取状态历史
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("获取状态历史")
    print("=" * 80)

    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 6:
            to_replay = state

    # -------------------------------------------------------------------------
    # 从检查点恢复
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("从检查点恢复")
    print("=" * 80)

    print("to_replay.next:", to_replay.next)
    print("to_replay.config:", to_replay.config)

    # -------------------------------------------------------------------------
    # 从某个时间点加载状态并重播
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("从某个时间点加载状态并重播")
    print("=" * 80)

    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # -------------------------------------------------------------------------
    # 【新功能演示】从指定检查点加载状态，然后提供新输入重新执行
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("【进阶演示】从早期检查点加载状态，提供新输入重新执行")
    print("=" * 80)

    early_checkpoint = None
    for state in graph.get_state_history(config):
        if len(state.values["messages"]) == 4:
            early_checkpoint = state
            break

    print(f"选择早期检查点：{len(early_checkpoint.values['messages'])} 条消息")
    print(f"检查点配置：{early_checkpoint.config}")
    print(f"该检查点的 next：{early_checkpoint.next}")
    print("-" * 80)

    print("\n提供新的用户输入（与之前不同）：")
    print("用户: 我想了解 LangGraph 的部署选项")
    print("-" * 80)

    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我想了解 LangGraph 的部署选项",
                },
            ],
        },
        early_checkpoint.config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # -------------------------------------------------------------------------
    # 演示另一种用法：从之前的状态继续执行
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("【另一种用法】恢复到之前的状态，提供新输入")
    print("=" * 80)

    checkpoints = list(graph.get_state_history(config))
    if len(checkpoints) >= 2:
        previous_checkpoint = checkpoints[1]
        print(f"选择倒数第二个检查点：{len(previous_checkpoint.values['messages'])} 条消息")
        print(f"该检查点的 next：{previous_checkpoint.next}")
        print("-" * 80)

        print("\n提供新的用户输入：")
        print("用户: 给我讲个笑话")
        print("-" * 80)

        events = graph.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "给我讲个笑话",
                    },
                ],
            },
            previous_checkpoint.config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()


if __name__ == '__main__':
    # run_base_chatbot()
    # run_add_tools_chatbot()
    # run_add_memory_chatbot()
    # run_add_human_in_the_loop_chatbot()
    run_time_travel_chatbot()
