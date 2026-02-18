"""
Raymond 核心推理模块

技术说明：
- 推理后端：Ollama 本地 API（/api/generate 端点）
- 为什么用 /api/generate 而不是 /api/chat：
    Qwen3 的 chat template 在 /api/chat 下会让模型继续自问自答，
    必须手动构造 prompt + stop tokens 才能正确截断
- Chat template：Qwen3 格式（<|im_start|> / <|im_end|>）
- Stop tokens：["<|im_end|>", "<|im_start|>"] 防止无限生成
- 支持多轮上下文（history 列表）
"""

import requests
import json
from pathlib import Path

# =================== 配置 ===================
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "raymond"

# 推理参数
INFERENCE_OPTIONS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.15,
    "num_predict": 150,           # 最多生成 150 个 token（约 50-100 汉字）
    "stop": ["<|im_end|>", "<|im_start|>"],  # 关键：防止模型自问自答
}

# Raymond 系统 prompt（从 persona 文件加载）
RESOURCES_DIR = Path(__file__).parent.parent / "resources"
PERSONA_FILE = RESOURCES_DIR / "raymond_persona.json"


def load_system_prompt() -> str:
    """从 persona 文件加载系统 prompt"""
    try:
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            persona = json.load(f)
        base = persona["system_prompt"]
    except (FileNotFoundError, KeyError):
        # fallback
        base = "你是Raymond，在美国留学的中国研究生。你在用微信和好朋友聊天。"

    # 附加关键说话规则（训练时用的 Modelfile 里也有）
    rules = """

【说话规则，必须严格遵守】
- 每条消息1-15个字，极短
- 用\\n分隔多条短消息，不要写长段落
- 绝对不能有*动作描写*，不能有旁白，不能有"温柔地"这种文艺词
- 不用书面语，不用标准标点，随意打字
- 不要自问自答，等对方回复

【口头禅】66、哈、f、说白了、不好说、俺、无敌了、我真谢了、我靠

【示例】
朋友：你在干嘛
你：打游戏哈\\n刚吃完饭

朋友：今天咋样
你：666还行\\n写了个bug

朋友：后悔出国吗
你：66后悔个锤子\\n天天后悔\\n但是回去不了"""

    return base + rules


# 全局加载（避免每次请求都读文件）
SYSTEM_PROMPT = load_system_prompt()


def build_prompt(history: list[dict], user_input: str) -> str:
    """
    构造 Qwen3 格式的 prompt

    Args:
        history: 历史消息列表，格式：[{"role": "user/assistant", "content": "..."}]
        user_input: 当前用户消息

    Returns:
        完整的 Qwen3 chat template 字符串
    """
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    for msg in history:
        role = msg["role"]  # "user" 或 "assistant"
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def chat(user_input: str, history: list[dict] = None) -> str:
    """
    核心推理函数

    Args:
        user_input: 用户输入文本
        history: 多轮对话历史（可选）

    Returns:
        Raymond 的回复文本
    """
    if history is None:
        history = []

    prompt = build_prompt(history, user_input)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": INFERENCE_OPTIONS,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "[错误] Ollama 未启动，请先运行 `ollama serve`"
    except requests.exceptions.Timeout:
        return "[错误] 推理超时，模型可能在加载中"
    except Exception as e:
        return f"[错误] {e}"


def check_ollama_status() -> bool:
    """检查 Ollama 服务是否正常运行"""
    try:
        response = requests.head(OLLAMA_BASE_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


# =================== 测试入口 ===================
if __name__ == "__main__":
    print("=== Raymond 推理模块测试 ===\n")

    if not check_ollama_status():
        print("❌ Ollama 未启动！请先运行: ollama serve")
        exit(1)

    print("✅ Ollama 正常运行\n")

    # 单轮测试
    test_inputs = [
        "你在干嘛",
        "今天吃啥",
        "美国生活咋样",
        "后悔出国吗",
        "来打游戏不",
    ]

    for msg in test_inputs:
        response = chat(msg)
        print(f"朋友：{msg}")
        print(f"Raymond：{response}")
        print()

    # 多轮测试
    print("--- 多轮对话测试 ---")
    history = []
    while True:
        user_input = input("朋友：").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        response = chat(user_input, history)
        print(f"Raymond：{response}\n")

        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # 保留最近 10 轮
        if len(history) > 20:
            history = history[-20:]
