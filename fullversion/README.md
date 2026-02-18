# Raymond LanguageModel — Full Version

> 用大模型蒸馏 + LoRA 微调，在本地部署一个模拟真实人物聊天风格的 AI，接入 QQ / 微信机器人。训练阶段使用 Claude API，**部署后推理零 API 成本**。

---

## 项目目标

- **训练期**：用 Claude Sonnet 作为「教师模型」蒸馏高质量训练数据，在 Qwen3-4B 上做 LoRA 微调
- **部署期**：模型量化为 GGUF 格式，通过 Ollama 在本地运行，对话成本为零
- **扩展性**：架构留有 Agent Memory（FAISS + LangGraph）接口，后续可继续迭代

---

## 整体流程

```
Step 1  数据蒸馏          Claude Sonnet API → 生成 1500 条风格训练数据
   ↓
Step 2  数据清洗          格式验证 + 长度过滤 + 风格校验 + MD5 去重 → 1495 条
   ↓
Step 3  LoRA 微调         Google Colab H100 / A100 / T4 + LLaMA-Factory
   ↓
Step 4  模型量化 & 部署   合并 LoRA → GGUF (Q4_K_M) → Ollama 本地服务
   ↓
Step 5  机器人接入        QQ (NoneBot2 + LLOneBot) / 微信 (WeChatFerry)
```

---

## 目录结构

```
fullversion/
├── data_pipeline/
│   ├── generate_distill_data.py   # Step 1: Claude API 数据蒸馏
│   └── clean_and_validate.py      # Step 2: 数据清洗与验证
│
├── training/
│   └── raymond_train.ipynb        # Step 3: Colab 训练笔记本
│
├── bot/
│   ├── raymond_core.py            # Step 4/5: Ollama 推理封装（核心）
│   ├── qq_bot.py                  # Step 5: QQ 机器人（NoneBot2）
│   ├── wechat_bot.py              # Step 5: 微信机器人（WeChatFerry）
│   └── DEPLOY.md                  # 部署详细步骤
│
├── resources/                     # ⚠️ 私密，不在 Git 中（见下方说明）
│   ├── raymond_persona.json       # 人格定义 & 系统 prompt
│   ├── raymond_fewshot.json       # 少样本示例（真实聊天记录）
│   └── raymond_memories.json      # 长期记忆（个人经历 & 关系）
│
├── Modelfile                      # Ollama 模型配置
├── pyproject.toml                 # 项目依赖（uv 管理）
└── .env.example                   # 环境变量模板（不含真实密钥）
```

> **⚠️ 注意**：`resources/`、`data_pipeline/raw_generated/`、`data_pipeline/processed/` 均含个人隐私数据，**不在 Git 仓库中**。

---

## 快速开始

### 前提条件

- Python 3.10 - 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器（或 pip）
- [Ollama](https://ollama.com) 已安装

### 1. 安装依赖

```bash
cd fullversion
uv sync
# 或使用 pip:
# pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 ANTHROPIC_API_KEY
```

### 3. 加载模型到 Ollama

模型文件托管在 HuggingFace，**先下载 GGUF 文件**：

```bash
# 方法 A：直接下载
huggingface-cli download RuimengLiu/raymond-gguf raymond-q4_k_m.gguf --local-dir .

# 方法 B：手动从 HuggingFace 下载后放到 fullversion/ 目录
# https://huggingface.co/RuimengLiu/raymond-gguf
```

然后导入到 Ollama：

```bash
ollama create raymond -f Modelfile
ollama list  # 确认 raymond 已创建
```

### 4. 测试推理

```bash
ollama serve  # 如果 Ollama 未在后台运行

python bot/raymond_core.py
```

### 5. 启动 QQ 机器人

```bash
# 先安装 LLOneBot（见 bot/DEPLOY.md）
python bot/qq_bot.py
```

---

## 技术详解

### Step 1 — 数据蒸馏

**核心思路**：用强模型（Claude Sonnet）扮演 Raymond，生成大量风格对话数据。

- 模型：`claude-sonnet-4-5-20250929`
- 数据量：1500 条 / 18 种场景（含权重采样）
- 格式：ShareGPT（LLaMA-Factory 标准格式）
- 关键参数：`max_tokens=2048`, `batch_size=5`, 支持断点续传

场景示例：`日常闲聊`, `游戏(铲)`, `吐槽美国生活`, `思念国内`, `科技话题`, `深夜哲学`, ...

### Step 2 — 数据清洗

清洗规则（通过率 99.7%）：
- 格式合法性检查（ShareGPT 结构）
- 对话长度：2–20 轮，单条消息 ≤ 2000 字符
- 风格标记：必须包含 Raymond 特征词（`66/哈/f/说白了` 等）
- MD5 去重，随机打散（seed=42）

### Step 3 — LoRA 微调

| 参数 | H100（≥70GB）| A100（≥35GB）| T4 |
|---|---|---|---|
| lora_rank | 64 | 32 | 16 |
| lora_alpha | 128 | 64 | 32 |
| batch_size | 8 | 4 | 2 |
| grad_accum | 2 | 4 | 8 |
| learning_rate | 5e-5 | 1e-4 | 1e-4 |
| 量化 | 无 | 无 | bfloat16 |

- 基础模型：`Qwen/Qwen3-4B-Instruct-2507`
- Chat template：`qwen3_nothink`（关闭 thinking token）
- 训练框架：LLaMA-Factory
- 4 epochs, warmup_steps=50, cosine scheduler
- 最终 loss：~1.15（健康范围 0.9–1.3）

### Step 4 — 量化 & 部署

```
HuggingFace merged_model
    → llama.cpp convert_hf_to_gguf.py (f16, ~8GB)
    → llama-quantize Q4_K_M (2.33GB, 约 75% 压缩)
    → ollama create raymond -f Modelfile
```

**关键推理配置**（`raymond_core.py`）：

```python
# 必须用 /api/generate 而非 /api/chat
# 原因：Qwen3 在 /api/chat 下会自问自答；
#       手动构造 chat template + stop tokens 才能正确截断
INFERENCE_OPTIONS = {
    "stop": ["<|im_end|>", "<|im_start|>"],  # 关键
    "temperature": 0.8,
    "repeat_penalty": 1.15,
    "num_predict": 150,
}
```

### Step 5 — 机器人接入

**QQ**：NoneBot2 + OneBot V11 适配器，兼容 LLOneBot / Lagrange
- 私聊：所有消息触发，保留最近 10 轮上下文
- 群聊：@机器人 触发，保留最近 5 轮上下文
- 重置命令：`重置` / `/reset`

**微信**：WeChatFerry（推荐，需 Windows 环境）/ itchat（备用）

---

## 模型文件

GGUF 模型文件体积 2.33GB，托管在 HuggingFace：

🤗 **[RuimengLiu/raymond-gguf](https://huggingface.co/RuimengLiu/raymond-gguf)**

| 文件 | 大小 | 说明 |
|---|---|---|
| `raymond-q4_k_m.gguf` | 2.33 GB | Q4_K_M 量化，推理用 |

---

## 关于 `resources/` 目录

`resources/` 包含 Raymond 的人格定义、真实聊天记录少样本、个人记忆，属于**个人隐私数据**，不在本仓库中。

如果你想自己训练一个类似的人物模型，参考以下结构创建：

```json
// raymond_persona.json
{
  "system_prompt": "你是[名字]，...[详细人物设定]...",
  "speaking_style": [...],
  "catchphrases": [...]
}

// raymond_fewshot.json
[
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]

// raymond_memories.json
{
  "background": "...",
  "relationships": {...},
  "recent_events": [...]
}
```

---

## 依赖

```toml
# 核心推理
ollama  # 本地运行（需单独安装）

# 数据蒸馏
anthropic>=0.40.0

# 机器人
nonebot2>=2.3.0
nonebot-adapter-onebot>=2.4.0

# Agent Memory（预留）
faiss-cpu, langchain, langgraph, sentence-transformers

# 微信（按需安装）
# wcferry      # Windows 专用
# itchat-uos   # 备用
```

---

## 后续计划

- [ ] Agent Memory：用 FAISS 存储长期记忆，让 Raymond 记住每个人的对话历史
- [ ] LangGraph 工作流：自动决定何时检索记忆、何时更新记忆
- [ ] 情绪状态管理：根据对话历史动态调整 Raymond 的「心情」
- [ ] 多模态：支持图片理解（转发 meme 等）

---

## License

MIT — 代码部分开源，训练数据 & 人格配置文件私密。
