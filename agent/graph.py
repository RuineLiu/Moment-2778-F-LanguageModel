"""
Raymond Agent - LangGraph 状态图
核心调度：system prompt + 长期记忆检索 + 短期对话历史 + LLM 生成 + 记忆更新
"""
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agent.prompt import build_system_prompt, build_fewshot_messages
from agent.memory import LongTermMemory
from agent.llm import create_llm
from agent.config import SHORT_TERM_WINDOW


class AgentState(TypedDict):
    """Agent 状态定义"""
    # 完整的消息历史（由 langgraph 的 add_messages reducer 管理）
    messages: Annotated[list[BaseMessage], add_messages]
    # 本轮检索到的长期记忆（每轮更新）
    retrieved_memories: list[str]
    # 是否需要提取新记忆
    should_extract_memory: bool


# 全局单例
_long_term_memory: LongTermMemory | None = None
_llm = None
_system_prompt: str | None = None
_fewshot_messages: list | None = None


def _get_memory() -> LongTermMemory:
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def _get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = build_system_prompt()
    return _system_prompt


def _get_fewshot() -> list:
    global _fewshot_messages
    if _fewshot_messages is None:
        _fewshot_messages = build_fewshot_messages()
    return _fewshot_messages


# ========== Graph Nodes ==========

def retrieve_memories(state: AgentState) -> dict:
    """节点1: 根据用户最新输入，检索相关的长期记忆"""
    messages = state["messages"]

    # 取最后一条 human 消息作为检索 query
    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"retrieved_memories": []}

    memory = _get_memory()
    results = memory.search(last_human)
    return {"retrieved_memories": results}


def generate_response(state: AgentState) -> dict:
    """节点2: 组装完整 prompt 并调用 LLM 生成回复"""
    llm = _get_llm()
    system_prompt = _get_system_prompt()
    fewshot = _get_fewshot()
    retrieved = state.get("retrieved_memories", [])

    # 1. 组装 system message（人设 + 记忆）
    full_system = system_prompt
    if retrieved:
        memory_text = "\n".join(f"- {m}" for m in retrieved)
        full_system += (
            f"\n\n[你的记忆 - 以下是你脑海中和当前话题相关的记忆碎片，"
            f"不用刻意提起每一条，只在对话自然需要的时候引用就好]\n{memory_text}"
        )

    # 2. 构建发送给 LLM 的消息列表
    prompt_messages = [SystemMessage(content=full_system)]

    # 3. 加入 few-shot 示例
    prompt_messages.extend(fewshot)

    # 4. 加入对话历史（短期记忆窗口）
    conversation = state["messages"]
    # 只保留最近 SHORT_TERM_WINDOW 轮
    max_msgs = SHORT_TERM_WINDOW * 2  # 每轮 = 1 human + 1 ai
    if len(conversation) > max_msgs:
        conversation = conversation[-max_msgs:]
    prompt_messages.extend(conversation)

    # 5. 调用 LLM
    response = llm.invoke(prompt_messages)

    return {
        "messages": [response],
        "should_extract_memory": True,
    }


def extract_memory(state: AgentState) -> dict:
    """节点3: 从本轮对话中提取值得记住的新事实"""
    if not state.get("should_extract_memory", False):
        return {}

    messages = state["messages"]
    if len(messages) < 2:
        return {"should_extract_memory": False}

    # 取最近一轮对话
    last_human = ""
    last_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not last_ai:
            last_ai = msg.content
        elif isinstance(msg, HumanMessage) and not last_human:
            last_human = msg.content
        if last_human and last_ai:
            break

    if not last_human or not last_ai:
        return {"should_extract_memory": False}

    # 用 LLM 判断是否有值得记住的新事实
    llm = _get_llm()
    extract_prompt = [
        SystemMessage(content=(
            "你是一个记忆提取器。分析以下对话，判断用户是否提到了值得长期记住的新事实"
            "（比如：个人信息、偏好、重要计划、重大事件、兴趣爱好等具体信息）。\n\n"
            "注意：\n"
            "- 只提取具体的、有信息量的事实，不要提取模糊的情绪或打招呼\n"
            "- 比如'用户今天心情不好'这种太泛，不值得记住\n"
            "- 但'用户正在准备考研'、'用户养了一只叫小白的猫'这种具体信息值得记住\n\n"
            "如果有，输出一个JSON数组，每个元素是一条事实字符串。\n"
            "如果没有值得记住的新信息，只输出空数组 []\n\n"
            "只输出JSON数组，不要输出任何其他内容。"
        )),
        HumanMessage(content=f"用户: {last_human}\nRaymond: {last_ai}"),
    ]

    try:
        result = llm.invoke(extract_prompt)
        content = result.content.strip()

        # 尝试解析 JSON
        import json
        # 处理可能的markdown代码块包裹
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        facts = json.loads(content)

        if isinstance(facts, list):
            memory = _get_memory()
            for fact in facts:
                if isinstance(fact, str) and len(fact) > 5:
                    memory.add_memory(fact)
    except (json.JSONDecodeError, Exception):
        pass

    return {"should_extract_memory": False}


# ========== Build Graph ==========

def build_graph() -> StateGraph:
    """构建 Raymond Agent 状态图"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("retrieve_memories", retrieve_memories)
    graph.add_node("generate_response", generate_response)
    graph.add_node("extract_memory", extract_memory)

    # 定义边
    graph.add_edge(START, "retrieve_memories")
    graph.add_edge("retrieve_memories", "generate_response")
    graph.add_edge("generate_response", "extract_memory")
    graph.add_edge("extract_memory", END)

    return graph


def create_agent():
    """创建并编译 Agent"""
    graph = build_graph()
    return graph.compile()
