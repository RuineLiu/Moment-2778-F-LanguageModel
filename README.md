# Raymond Agent - Local Version

> **v0.2.0** | Memory-Augmented Persona Chatbot

Raymond Agent 是一个基于记忆增强的人格聊天机器人。它通过 System Prompt + Few-shot + 双层记忆系统（短期 + 长期），让大语言模型扮演一个从真实群聊记录中提炼出的角色 "Raymond"，并能在多轮对话中动态积累记忆。

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  LangGraph Agent                │
│                                                 │
│  ┌──────────────┐   ┌──────────────────────┐    │
│  │   用户输入    │──▶│  1. retrieve_memories │    │
│  └──────────────┘   │  FAISS 向量检索 top-K  │    │
│                     └──────────┬───────────┘    │
│                                ▼                │
│  ┌──────────────────────────────────────────┐   │
│  │         2. generate_response              │   │
│  │  System Prompt (人设 + 检索到的记忆)       │   │
│  │  + Few-shot 示例 (10组真实对话)            │   │
│  │  + 短期记忆 (最近N轮对话历史)              │   │
│  │  → 调用 LLM API 生成回复                   │   │
│  └──────────────────┬───────────────────────┘   │
│                     ▼                           │
│  ┌──────────────────────────────────────────┐   │
│  │         3. extract_memory                 │   │
│  │  LLM 分析本轮对话 → 提取新事实 → 存入FAISS │   │
│  └──────────────────┬───────────────────────┘   │
│                     ▼                           │
│               输出 Raymond 的回复                │
└─────────────────────────────────────────────────┘
```

**双层记忆系统：**

| 类型 | 实现 | 作用 | 生命周期 |
|------|------|------|----------|
| 短期记忆 | 对话历史滑动窗口 | 维持多轮对话上下文 | 随对话session结束而清空 |
| 长期记忆 | FAISS 向量数据库 | 存储跨session的事实性信息 | 持久化到磁盘，跨session保留 |

---

## Project Structure

```
.
├── agent/                     # 核心 Agent 模块
│   ├── __init__.py
│   ├── config.py              # 配置管理（环境变量 + 默认值）
│   ├── graph.py               # LangGraph 状态图（三节点流水线）
│   ├── llm.py                 # LLM 工厂（Anthropic / OpenAI / Ollama）
│   ├── memory.py              # 长期记忆（FAISS + Embedding）
│   └── prompt.py              # System Prompt + Few-shot 构建
├── resources/                 # 静态资源
│   ├── raymond_persona.json   # 人格画像（口头禅、句式、性格等）
│   ├── raymond_fewshot.json   # 10组真实对话 Few-shot 示例
│   └── raymond_memories.json  # 42条初始长期记忆
├── chat.py                    # 命令行聊天入口
├── .env                       # API Key 配置（不入版本控制）
└── pyproject.toml             # 项目依赖
```

---

## Quick Start

### 1. Requirements

- Python 3.10 - 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Anthropic API Key (default), or OpenAI / Ollama

### 2. Install

```bash
# clone 并进入项目目录
cd <project-root>

# 用 uv 安装依赖
uv sync
```

### 3. Configure

创建 `.env` 文件：

```bash
# 必填 - Anthropic (默认)
ANTHROPIC_API_KEY=your-api-key-here

# 或者用 OpenAI
# RAYMOND_LLM_PROVIDER=openai
# OPENAI_API_KEY=your-openai-key-here

# 或者用 Ollama (本地模型，无需API key)
# RAYMOND_LLM_PROVIDER=ollama
# RAYMOND_LLM_MODEL=qwen3:32b
```

### 4. Run

```bash
uv run python chat.py
```

首次运行会自动下载 embedding 模型 (`BAAI/bge-small-zh-v1.5`, ~90MB) 并从 `raymond_memories.json` 构建 FAISS 索引。

---

## Chat Commands

| Command | Description |
|---------|-------------|
| `/memory` | 查看长期记忆总条数 |
| `/search <keyword>` | 在长期记忆中语义搜索 |
| `/clear` | 清空当前对话历史（长期记忆保留） |
| `exit` / `quit` / `q` | 退出程序 |

---

## Configuration

所有配置通过环境变量管理，均有默认值：

| Variable | Default | Description |
|----------|---------|-------------|
| `RAYMOND_LLM_PROVIDER` | `anthropic` | LLM 提供方: `anthropic` / `openai` / `ollama` |
| `RAYMOND_LLM_MODEL` | (auto) | 模型名称，留空则使用各 provider 默认值 |
| `RAYMOND_TEMPERATURE` | `0.8` | 生成温度 |
| `RAYMOND_EMBED_PROVIDER` | `local` | Embedding 提供方: `local` / `ollama` / `openai` |
| `RAYMOND_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding 模型 |
| `RAYMOND_SHORT_TERM_WINDOW` | `20` | 短期记忆保留的对话轮数 |
| `RAYMOND_LONG_TERM_TOP_K` | `5` | 每轮检索的长期记忆条数 |
| `RAYMOND_FAISS_PATH` | `raymond_memory_index` | FAISS 索引存储路径 |

**Default models per provider:**

| Provider | Default Model |
|----------|--------------|
| Anthropic | `claude-sonnet-4-5-20250929` |
| OpenAI | `gpt-4o` |
| Ollama | `qwen3:32b` |

---

## How It Works

### Data Pipeline

原始数据是一份4人微信群聊记录 (13,974条消息)，从中提炼三种资源：

1. **Persona** (`raymond_persona.json`) - 从约1000条消息中归纳出 Raymond 的说话风格、口头禅、句式习惯、性格特征
2. **Few-shot Examples** (`raymond_fewshot.json`) - 10组真实对话覆盖不同场景（闲聊、吐槽、自嘲、游戏、认真倾诉等）
3. **Initial Memories** (`raymond_memories.json`) - 42条客观事实（个人信息、留学经历、社交关系等）

### Per-Turn Flow

每次用户发送消息，Agent 按顺序执行三个节点：

1. **retrieve_memories** - 用用户消息在 FAISS 中做语义搜索，返回 top-K 条最相关的长期记忆
2. **generate_response** - 将 System Prompt + 检索记忆 + Few-shot + 近N轮对话历史 组装成完整 prompt，调用 LLM 生成回复
3. **extract_memory** - 用 LLM 分析本轮对话，提取新事实存入 FAISS（只提取具体的、有信息量的事实）

### Prompt Assembly

发送给 LLM 的消息结构：

```
[SystemMessage]     ← 人设描述 + 口头禅/句式规则 + 本轮检索到的记忆碎片
[HumanMessage]      ← Few-shot example 1 (user)
[AIMessage]         ← Few-shot example 1 (raymond)
[HumanMessage]      ← Few-shot example 2 (user)
[AIMessage]         ← Few-shot example 2 (raymond)
...                 ← (共10组 few-shot)
[HumanMessage]      ← 对话历史 turn 1
[AIMessage]         ← 对话历史 turn 1
...                 ← (最近 SHORT_TERM_WINDOW 轮)
[HumanMessage]      ← 当前用户输入
```

---

## Example

```
========================================
  Raymond Agent
  模型: anthropic / claude-sonnet-4-5-20250929
========================================

你: 在吗
Raymond: 在哈
咋了

你: 你那边现在几点
Raymond: 晚上八点多哈
怎么了米粒

你: 还没睡 在干嘛
Raymond: 刚铲完
正准备洗澡睡觉哈
你那边几点了

你: 哈哈哈你真是个宅男
Raymond: 66666666
说白了
俺就是宅男
宿舍无敌了
每天就是上课铲睡觉
我真谢了
```

---

## Prior Work

本项目从 LoRA 微调方案迁移而来。v0.1 使用 LLaMA-Factory 对 Qwen3-4B 做 LoRA 微调，由于训练数据质量和小模型能力限制，效果不理想。v0.2 转为 Prompt Engineering + Memory Agent 架构。

- **Working Log**: https://docs.google.com/document/d/1jJH5MEQr-DJ2kngQjkMcTK_P2J_6gUesXCWpDYb5sDQ/edit?usp=drive_link
- **Training Code (v0.1)**: https://colab.research.google.com/drive/1EBZN1sDVm7b5sS6WTTq0mSvYe7C2cwx0?usp=sharing
- **Huggingface (v0.1)**: https://huggingface.co/RuimengLiu/raymond-qwen3-4b-lora
