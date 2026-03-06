# Agent工程师技术栈分类

## 一、Agent框架/架构

| 技术 | 用途 |
|------|------|
| **LangGraph** | 构建有状态、多步骤的Agent工作流，基于LangChain |
| **AutoGen** | 微软开源的多Agent协作框架，支持自定义Agent和对话模式 |
| **OpenManus** | Meta开源的多Agent系统，用于复杂任务自动化 |
| **LangFlow** | LangChain的可视化工作流编排工具，低代码构建Agent |
| **Multi-agent** | 多Agent系统设计模式，Agent间通信与协作机制 |
| **OWL** | Google's OWL 多Agent框架 |

## 二、长链路Agent机制

| 技术 | 用途 |
|------|------|
| **长链路Agent机制** | 处理复杂任务的多步骤规划、反思、工具调用链 |
| **ReAct** | 推理+行动的Agent范式，结合思考与工具使用 |

## 三、大语言模型/RAG

| 技术 | 用途 |
|------|------|
| **LLM/RAG基本原理** | 大语言模型基础 + 检索增强生成技术 |
| **VLLM** | 高性能LLM推理加速框架，PagedAttention技术 |
| **Prompt调试与上下文构建经验** | Prompt工程技巧，上下文窗口管理 |

## 四、模型训练/微调

| 技术 | 用途 |
|------|------|
| **LoRA** | 低秩适配微调技术，降低微调成本 |
| **QLoRA** | 量化的LoRA，进一步降低显存需求 |
| **LLaMA-Factory** | 通用大模型微调框架 |

## 五、协议/工具集成

| 技术 | 用途 |
|------|------|
| **MCP** | Model Context Protocol，Anthropic推出的Agent标准化协议 |
| **A2A** | Agent-to-Agent Protocol，Google主导的Agent通信协议 |
| **skills** | Agent工具/技能定义与调用机制 |

---

## 六、向量数据库

| 技术 | 用途 |
|------|------|
| **Chroma** | 开源向量数据库 |
| **Milvus** | 大规模向量数据库 |
| **Pinecone** | 云端向量数据库服务 |
| **Qdrant** | 高性能向量搜索引擎 |

## 七、其他Agent开发框架

| 技术 | 用途 |
|------|------|
| **LangChain** | Agent应用开发基础框架 |
| **CrewAI** | 多Agent协作框架 |
| **SmolAgents** | 轻量级Agent框架 |

## 八、LLM API/部署

| 技术 | 用途 |
|------|------|
| **OpenAI API** | GPT系列模型接口 |
| **Anthropic API** | Claude系列模型接口 |
| **Ollama** | 本地LLM运行工具 |

## 九、工具/搜索

| 技术 | 用途 |
|------|------|
| **Tavily** | Agent专用搜索API |
| **DuckDuckGo** | 搜索引擎工具 |
| **SerpAPI** | Google搜索API |

## 十、监控/可观测性

| 技术 | 用途 |
|------|------|
| **LangSmith** | LangChain应用监控调试平台 |
| **OpenTelemetry** | 可观测性标准 |

## 十一、云服务/部署

| 技术 | 用途 |
|------|------|
| **Docker** | 容器化部署 |
| **Kubernetes** | 容器编排 |
| **AWS/GCP/Azure** | 云平台服务 |
