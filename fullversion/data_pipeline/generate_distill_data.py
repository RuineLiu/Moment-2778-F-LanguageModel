"""
Step 1: 数据蒸馏脚本
用 Claude API 批量生成高质量的 Raymond 风格训练对话数据

技术说明：
- 方法：LLM 数据蒸馏（Data Distillation / Synthetic Data Generation）
  即用一个强大的"教师模型"（Claude）生成训练数据，再用来训练一个较小的"学生模型"
- API：Anthropic Claude API（claude-sonnet-4-5）
- 输出格式：ShareGPT JSON（LLaMA-Factory 标准训练格式）
- 生成策略：覆盖多种话题场景，每个场景生成多条，保证多样性
"""

import json
import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

# 加载 .env 文件（在 fullversion 目录下）
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# =================== 路径配置 ===================
BASE_DIR = Path(__file__).parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
OUTPUT_DIR = Path(__file__).parent / "raw_generated"
OUTPUT_FILE = OUTPUT_DIR / "distilled_data.json"

PERSONA_FILE = RESOURCES_DIR / "raymond_persona.json"
FEWSHOT_FILE = RESOURCES_DIR / "raymond_fewshot.json"

# =================== 生成配置 ===================
TARGET_TOTAL = 1500       # 目标生成总条数
BATCH_SIZE = 5            # 每次 API 调用生成的对话条数（控制单次 token 消耗）
DELAY_BETWEEN_CALLS = 1.0 # 每次 API 调用之间的间隔（秒），避免触发限流

# =================== 话题场景定义 ===================
# 每个场景定义：(描述, 权重)
# 权重决定该场景被采样的概率，越高生成越多
SCENARIOS = [
    # 日常闲聊类
    ("两个好朋友的日常闲聊，话题随机，比如今天干了什么、吃了什么、有什么新鲜事", 15),
    ("深夜聊天，两人都睡不着，聊一些没营养但很舒服的话", 8),
    ("早上刚起床，聊早餐、天气、今天的计划", 6),

    # 吐槽类
    ("吐槽室友、邻居或者陌生人的奇葩行为，越无语越好", 10),
    ("吐槽美国生活的不便、文化差异，想念国内的东西", 10),
    ("吐槽作业多、考试难、教授讲得烂，学业压力", 8),

    # 游戏类
    ("叫朋友一起打游戏（'铲'），聊游戏里的事，炫耀战绩或者吐槽队友", 10),
    ("聊某个手游、网游的内容，攻略、皮肤、角色", 6),

    # 技术/学习类
    ("聊代码、bug、编程作业，用自嘲或骄傲的方式描述自己的技术", 8),
    ("讨论 AI 工具（GPT、Claude、Cursor 等），分享使用体验", 6),
    ("期中/期末考试前后，聊备考、成绩、GPA", 8),

    # 做饭类
    ("自己做了饭，结果翻车了或者出乎意料地好吃，聊做饭经过", 8),
    ("讨论想吃的东西、国内美食、想念某道菜", 6),

    # 感情/情绪类
    ("聊到感情话题，Raymond 从调侃慢慢变认真，说出真实的孤独感或想法", 7),
    ("安慰朋友，Raymond 先插科打诨然后认真说两句鼓励的话", 6),
    ("聊到留学孤独、异乡生活，情绪比较复杂", 6),

    # 朋友互动类
    ("和樊明磊（米粒）的互动，互相损对方，叫宝宝，聊八卦", 10),
    ("朋友发来一个有趣的视频或图片，Raymond 评论", 5),
    ("朋友问 Raymond 要建议，Raymond 给出又毒又有道理的回答", 8),

    # 健身类
    ("聊健身、去健身房的事，偶尔炫一下，偶尔偷懒找借口", 5),
]

# 预计算权重列表
SCENARIO_TEXTS = [s[0] for s in SCENARIOS]
SCENARIO_WEIGHTS = [s[1] for s in SCENARIOS]


def load_persona() -> dict:
    with open(PERSONA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fewshot() -> dict:
    with open(FEWSHOT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_teacher_system_prompt(persona: dict, fewshot: dict) -> str:
    """
    构建教师模型的 system prompt
    核心思路：让 Claude 完全扮演 Raymond，生成"真实"的对话
    而不是让 Claude 以"生成训练数据"的视角创作——这样输出更自然
    """
    # 完整注入 persona
    p = persona
    sp = p["system_prompt"]

    prompt = f"""你是一个训练数据生成器。你的任务是生成高质量的对话训练样本，用于训练一个叫 Raymond 的聊天机器人。

Raymond 的完整人设如下：

{sp}

补充细节：
- 身份：{p['basic_info']['身份']}，{p['basic_info']['所在地']}
- 家庭：{p['basic_info']['家庭']}
- 日常：{p['basic_info']['日常']}
- 好友：{p['basic_info']['社交圈']}

说话风格（非常重要）：
- 总体特征：{p['speaking_style']['总体特征']}
- 口头禅：{', '.join(p['speaking_style']['口头禅'][:6])}
- 句式习惯：{'; '.join(p['speaking_style']['句式习惯'][:4])}
- 回复长度：大多数 1-15 个字，像微信聊天一样短

性格特点：
- 幽默方式：{p['personality_traits']['幽默方式']}
- 社交模式：{p['personality_traits']['社交模式']}
- 情感特点：{p['personality_traits']['情感特点']}

真实风格示例（从真实群聊记录中提取）：
"""
    # 注入 few-shot 示例
    for example in fewshot["examples"][:8]:  # 取前8个覆盖主要风格
        prompt += f"\n[{example['topic']} - {example['style_note']}]\n"
        for turn in example["conversation"]:
            role = "朋友" if turn["role"] == "human" else "Raymond"
            prompt += f"{role}: {turn['content']}\n"

    prompt += """
生成规则：
1. Raymond 的回复必须短而碎，像真实微信聊天，用 \\n 分隔多条消息
2. 保持人物一致性：真实、不做作、有时懒、有时真诚
3. 不要让 Raymond 说教、不要书面语、不要完美的标点
4. 对话要自然，不是所有话题都要以正能量结尾
5. 朋友那方可以是任何人（不一定是樊明磊），要多样化
"""
    return prompt


def build_generation_request(scenario: str, batch_size: int) -> str:
    """构建单次生成请求的 user prompt"""
    return f"""请生成 {batch_size} 条对话训练样本。

话题场景：{scenario}

输出格式要求（严格遵守）：
- 输出一个 JSON 数组
- 每个元素是一个对话对象，包含 "conversations" 字段
- "conversations" 是消息列表，每条消息有 "from"（"human" 或 "gpt"）和 "value" 字段
- Raymond 对应 "gpt"，朋友对应 "human"
- 每条对话 3-12 轮，Raymond 的回复要体现真实说话风格

示例格式：
[
  {{
    "conversations": [
      {{"from": "human", "value": "你在干嘛"}},
      {{"from": "gpt", "value": "打游戏哈\\n躺着摆烂"}},
      {{"from": "human", "value": "作业不写吗"}},
      {{"from": "gpt", "value": "明天再说哈\\n今天额度用完了"}}
    ]
  }}
]

只输出 JSON 数组，不要任何其他文字。"""


def parse_response(content: str) -> list:
    """解析 Claude 返回的 JSON，容错处理"""
    content = content.strip()

    # 去掉 markdown 代码块
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # 尝试找到 JSON 数组的开始和结束
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return []


def validate_sample(sample: dict) -> bool:
    """校验单条样本的基本格式"""
    if not isinstance(sample, dict):
        return False
    convs = sample.get("conversations", [])
    if not isinstance(convs, list) or len(convs) < 2:
        return False
    for msg in convs:
        if not isinstance(msg, dict):
            return False
        if msg.get("from") not in ("human", "gpt"):
            return False
        if not msg.get("value", "").strip():
            return False
    return True


def add_system_prompt(samples: list, system_prompt: str) -> list:
    """给每条样本注入完整的 system prompt（训练时用）"""
    result = []
    for s in samples:
        new_s = dict(s)
        # 在 conversations 最前面插入 system 消息
        new_s["conversations"] = [
            {"from": "system", "value": system_prompt}
        ] + list(s.get("conversations", []))
        result.append(new_s)
    return result


def load_existing(output_file: Path) -> list:
    """加载已有数据（支持断点续传）"""
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"检测到已有数据：{len(data)} 条，继续追加生成")
        return data
    return []


def save_data(data: list, output_file: Path):
    """保存数据"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_raymond_system_prompt(persona: dict) -> str:
    """构建注入训练数据的 system prompt（给学生模型用的）"""
    p = persona
    sp = p["system_prompt"]

    prompt = sp + "\n\n你的口头禅包括：\n"
    for cp in p["speaking_style"]["口头禅"]:
        prompt += f"- {cp}\n"

    prompt += "\n你的句式习惯：\n"
    for h in p["speaking_style"]["句式习惯"]:
        prompt += f"- {h}\n"

    prompt += "\n重要规则：\n"
    prompt += "- 和你聊天的是你的好朋友，直接像老朋友一样聊\n"
    prompt += "- 回复要像微信聊天，用\\n分隔多条短消息\n"
    prompt += "- 绝大多数时候每条消息控制在1-15个字\n"
    prompt += "- 不要用书面语，不要用标准标点，像真人打字一样随意\n"
    prompt += "- 不要主动解释自己是AI\n"

    return prompt


def main():
    print("=" * 50)
    print("Raymond 数据蒸馏脚本")
    print("=" * 50)

    # 检查 API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 ANTHROPIC_API_KEY\n"
            "export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = Anthropic(api_key=api_key)

    # 加载资源
    print("加载 persona 和 fewshot 资源...")
    persona = load_persona()
    fewshot = load_fewshot()

    # 构建 prompt
    teacher_system = build_teacher_system_prompt(persona, fewshot)
    raymond_system = build_raymond_system_prompt(persona)

    # 加载已有数据（断点续传）
    all_data = load_existing(OUTPUT_FILE)
    already_count = len(all_data)

    print(f"目标生成：{TARGET_TOTAL} 条")
    print(f"当前已有：{already_count} 条")
    print(f"还需生成：{max(0, TARGET_TOTAL - already_count)} 条")
    print()

    call_count = 0
    fail_count = 0

    while len(all_data) < TARGET_TOTAL:
        remaining = TARGET_TOTAL - len(all_data)
        batch = min(BATCH_SIZE, remaining)

        # 随机采样话题场景（加权）
        scenario = random.choices(SCENARIO_TEXTS, weights=SCENARIO_WEIGHTS, k=1)[0]

        print(f"[{len(all_data)}/{TARGET_TOTAL}] 场景: {scenario[:30]}...", end=" ", flush=True)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                system=teacher_system,
                messages=[
                    {"role": "user", "content": build_generation_request(scenario, batch)}
                ]
            )
            call_count += 1

            content = response.content[0].text
            samples = parse_response(content)

            # 校验并注入 system prompt
            valid_samples = [s for s in samples if validate_sample(s)]
            valid_samples = add_system_prompt(valid_samples, raymond_system)

            all_data.extend(valid_samples)
            print(f"✓ 获得 {len(valid_samples)} 条（有效率 {len(valid_samples)}/{len(samples)}）")

            # 每 10 次调用保存一次（防止意外中断丢失数据）
            if call_count % 10 == 0:
                save_data(all_data, OUTPUT_FILE)
                print(f"  → 已保存 {len(all_data)} 条到文件")

            time.sleep(DELAY_BETWEEN_CALLS)

        except Exception as e:
            fail_count += 1
            print(f"✗ 失败: {e}")
            time.sleep(3)  # 出错后等待更长时间

            if fail_count > 10:
                print("连续失败次数过多，中止生成")
                break

    # 最终保存
    save_data(all_data, OUTPUT_FILE)

    print()
    print("=" * 50)
    print(f"生成完成！")
    print(f"总条数：{len(all_data)}")
    print(f"API 调用次数：{call_count}")
    print(f"失败次数：{fail_count}")
    print(f"输出文件：{OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
