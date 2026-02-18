"""
LLM 初始化模块 - 根据配置创建对应的 ChatModel
"""
from langchain_core.language_models.chat_models import BaseChatModel
from agent.config import LLM_PROVIDER, LLM_TEMPERATURE, get_model_name


def create_llm() -> BaseChatModel:
    """根据配置创建 LLM 实例"""
    model_name = get_model_name()

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=LLM_TEMPERATURE,
            max_tokens=512,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=LLM_TEMPERATURE,
            max_tokens=512,
        )
    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=LLM_TEMPERATURE,
            num_ctx=8192,
        )
    else:
        raise ValueError(f"不支持的 LLM provider: {LLM_PROVIDER}")
