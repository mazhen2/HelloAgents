# 创建一个提示词模板
from cgitb import text
from email import message
from os import name
import langchain_core


def create_joke_prompt_template():
    """创建一个讲笑话的提示词模板"""
    from langchain_core.prompts import PromptTemplate

    prompts_template = PromptTemplate.from_template(
        "给我讲一个关于{content}的{adjective}笑话"
    )

    result = prompts_template.format(
        adjective="冷", content="猴子"
    )
    print(result)
    return result


def create_multi_turn_chat_template():
    """创建一个多轮对话的聊天提示词模板"""
    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位人工智能助手，你的名字是{name}"),
            ("human", "你好"),
            ("ai", "我很好，谢谢！"),
            ("human", "{user_input}")
        ]
    )

    messages = chat_template.format_messages(
        name="Bob", user_input="你的名字叫什么"
    )
    print(messages)
    return messages


def create_assistant_chat_template():
    """创建一个助手对话的聊天提示词模板"""
    from langchain_core.prompts import HumanMessagePromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="你是一位乐于助人的助手，可以润色内容"
            ),
            HumanMessagePromptTemplate.from_template(
                "{text}"
            )
        ]
    )

    message = chat_template.format_messages(
        text="我晚上不想吃东西"
    )
    print(message)
    return message


def create_chat_with_messages_placeholder():
    """创建一个包含消息占位符的聊天提示词模板"""
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate

    # prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful assistant"),
    #         MessagesPlaceholder("msgs")
    #     ]
    # )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{msgs}")
        ]
    )

    messages = prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
    print(messages)
    return messages


def create_few_shot_prompt_with_examples():
    """使用示例集构建小样本提示词（Few-Shot Prompting）

    对应教程：创建示例集 -> 创建示例格式化器 -> 合并为 FewShotPromptTemplate
    """
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

    # 1. 创建示例集
    examples = [
        {
            "question": "我的狗叫什么名字，它的眼睛是棕色的还是黑色的？",
            "answer": "我的狗叫Bob，它的眼睛是黑色的。",
        },
        {
            "question": "craigslist的创始人是谁？",
            "answer": "craigslist的创始人是Craig Newmark。Craig Newmark于1995年创立craigslist。",
        },
        {
            "question": "Mary Mall washington的父亲是谁？",
            "answer": "Mary Mall washington的父亲是Joseph Mall。",
        },
        {
            "question": "(大白鲨)和(星球大战)的导演是同一个人吗？",
            "answer": "不是。大白鲨的导演是Steven Spielberg，星球大战的导演是Martin Campbell。",
        },
    ]

    # 2. 创建小样本示例的格式化程序
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Question: {question}\nAnswer: {answer}",
    )
    # 演示格式化单个示例
    print("单个示例格式化:", example_prompt.format(**examples[0]))

    # 3. 将示例和格式化程序合并为 FewShotPromptTemplate
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Input: {input}",
        input_variables=["input"],
    )
    result = prompt.format(input="你喜欢吃什么水果？")
    print("Few-Shot 提示词（示例集）:\n", result)
    return result


def create_few_shot_prompt_with_selector():
    """使用示例选择器构建小样本提示词（语义相似度选择）

    当示例数量很大时，用 ExampleSelector 按输入智能选取最相关示例。
    """
    # 导入语义相似度示例选择器
    from langchain_core.example_selectors.semantic_similarity import (
        SemanticSimilarityExampleSelector,
    )
    # 导入小样本提示词模板和基础提示词模板
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

    # 定义问答示例列表，每个示例包含问题和答案
    examples = [
        {
            "question": "我的狗叫什么名字，它的眼睛是棕色的还是黑色的？",
            "answer": "我的狗叫 Bob，它的眼睛是黑色的。",
        },
        {
            "question": "craigslist 的创始人是谁？",
            "answer": "craigslist 的创始人是 Craig Newmark。Craig Newmark 于 1995 年创立 craigslist。",
        },
        {
            "question": "Mary Mall washington 的父亲是谁？",
            "answer": "Mary Mall washington 的父亲是 Joseph Mall。",
        },
        {
            "question": "(大白鲨) 和 (星球大战) 的导演是同一个人吗？",
            "answer": "不是。大白鲨的导演是 Steven Spielberg，星球大战的导演是 Martin Campbell。",
        },
    ]

    # 创建示例提示词模板，定义每个示例的展示格式
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],  # 指定模板需要的变量
        template="Question: {question}\nAnswer: {answer}",  # 示例的显示格式
    )

    try:
        # 导入嵌入模型和向量存储库
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        # 使用语义相似度选择器：按输入选取最相似的 k 个示例
        # from_examples 方法会：
        # 1. 使用 HuggingFaceEmbeddings 将所有示例转换为向量
        # 2. 存储在 Chroma 向量数据库中
        # 3. 当有新输入时，计算输入向量并找出最相似的 k 个示例
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=examples,  # 示例列表
            embeddings=HuggingFaceEmbeddings(),  # 嵌入模型，用于将文本转换为向量
            vectorstore_cls=Chroma,  # 向量存储类，用于存储和检索向量
            k=2,  # 每次选择最相似的 2 个示例
        )

        # 创建小样本提示词模板
        prompt = FewShotPromptTemplate(
            example_selector=example_selector,  # 示例选择器，动态选择相关示例
            example_prompt=example_prompt,  # 单个示例的格式化模板
            suffix="Input: {input}",  # 提示词的后缀部分，包含用户实际输入
            input_variables=["input"],  # 整个提示词模板需要的输入变量
        )

        # 格式化提示词，自动选择与输入最相关的 2 个示例
        result = prompt.format(input="你喜欢吃什么水果？")
        print("Few-Shot 提示词（示例选择器）:\n", result)
        return result

    except ImportError as e:
        # 捕获导入错误，提示用户需要安装相关依赖
        print(
            "使用示例选择器需要安装：pip install langchain_community chromadb sentence-transformers\n"
            f"ImportError: {e}"
        )
        return None


if __name__ == "__main__":
    # create_joke_prompt_template()
    # create_multi_turn_chat_template()
    # create_assistant_chat_template()
    # create_chat_with_messages_placeholder()
    # create_few_shot_prompt_with_examples()
    create_few_shot_prompt_with_selector()
