"""
Raymond Agent 配置
"""
import os

# LLM 配置 - 通过环境变量切换 provider
# 支持: "anthropic", "openai", "ollama"
LLM_PROVIDER = os.getenv("RAYMOND_LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("RAYMOND_LLM_MODEL", "")  # 留空则使用各provider默认值
LLM_TEMPERATURE = float(os.getenv("RAYMOND_TEMPERATURE", "0.8"))

# 各 provider 的默认模型
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "ollama": "qwen3:32b",
}

# Embedding 配置
# 默认用本地 sentence-transformers，不依赖 Ollama
EMBED_PROVIDER = os.getenv("RAYMOND_EMBED_PROVIDER", "local")
EMBED_MODEL = os.getenv("RAYMOND_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

# 记忆配置
SHORT_TERM_WINDOW = int(os.getenv("RAYMOND_SHORT_TERM_WINDOW", "20"))  # 保留最近N轮对话
LONG_TERM_TOP_K = int(os.getenv("RAYMOND_LONG_TERM_TOP_K", "5"))  # 检索top-K条长期记忆
FAISS_INDEX_PATH = os.getenv("RAYMOND_FAISS_PATH", "raymond_memory_index")

# 资源文件路径
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
PERSONA_FILE = os.path.join(RESOURCES_DIR, "raymond_persona.json")
FEWSHOT_FILE = os.path.join(RESOURCES_DIR, "raymond_fewshot.json")
MEMORIES_FILE = os.path.join(RESOURCES_DIR, "raymond_memories.json")


def get_model_name() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    return DEFAULT_MODELS.get(LLM_PROVIDER, "claude-sonnet-4-5-20250929")
