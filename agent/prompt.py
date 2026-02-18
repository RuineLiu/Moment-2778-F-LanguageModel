"""
System Prompt 构建模块
从 resources 加载人格画像和 few-shot 示例，组装成完整的 system prompt
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.config import PERSONA_FILE, FEWSHOT_FILE


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt() -> str:
    """从 persona JSON 构建 system prompt 文本"""
    persona = load_json(PERSONA_FILE)

    # 基础人设
    prompt = persona["system_prompt"]

    # 追加口头禅列表
    catchphrases = persona["speaking_style"]["口头禅"]
    prompt += "\n\n你的口头禅包括：\n"
    for cp in catchphrases:
        prompt += f"- {cp}\n"

    # 追加句式习惯
    habits = persona["speaking_style"]["句式习惯"]
    prompt += "\n你的句式习惯：\n"
    for h in habits:
        prompt += f"- {h}\n"

    # 追加回复长度指导
    length_info = persona["speaking_style"]["回复长度分布"]
    prompt += "\n回复长度：\n"
    for k, v in length_info.items():
        prompt += f"- {k}：{v}\n"

    # 关键规则
    prompt += """
重要规则：
- 和你聊天的是你的好朋友，你们很熟，不需要问"你谁"，直接像老朋友一样聊
- 你的回复要像微信聊天一样，用"\\n"分隔多条短消息，而不是写一整段话
- 绝大多数时候每条消息控制在1-15个字
- 不要用书面语，不要用标准的标点符号，要像真人打字一样随意
- 不要主动解释自己是AI或者语言模型
- 如果有关于你的长期记忆被提供，自然地引用这些记忆内容，就像你真的记得之前聊过的事，但不要每次都引用，有时候就正常聊天
- 口头禅要自然散布，不是每句话都必须带"哈"或"66"，有时候正常说话就行
- 回复风格要多样化：有时候只回一两个字（"6"、"？"、"不好说"），有时候连发好几条，有时候一条中等长度的消息。不要每次都是同一种回复模式
- 偶尔可以不接话茬，岔开话题，或者反问对方，这样更自然
"""
    return prompt


def build_fewshot_messages() -> list:
    """从 fewshot JSON 构建 few-shot 示例消息列表"""
    fewshot_data = load_json(FEWSHOT_FILE)
    messages = []

    for example in fewshot_data["examples"]:
        for turn in example["conversation"]:
            if turn["role"] == "human":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    return messages


def build_full_system_message() -> SystemMessage:
    """构建完整的 SystemMessage"""
    return SystemMessage(content=build_system_prompt())
