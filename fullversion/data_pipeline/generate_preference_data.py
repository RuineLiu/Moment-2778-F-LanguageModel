"""
Step 1b: DPO / 奖励模型 偏好数据生成脚本
生成 (chosen, rejected) 对话对，用于 DPO 训练或奖励模型训练

技术说明：
- DPO (Direct Preference Optimization)：无需奖励模型，直接对比 chosen/rejected 训练
  优点：稳定、简单、不需要 PPO 调参
  缺点：无法在线采样，泛化能力弱于 PPO
- RLHF / PPO：用本脚本数据训练奖励模型，再用 PPO 强化对齐
  优点：可以在线探索，对齐质量更高
  缺点：训练复杂，需要调两套超参
- 本脚本输出：LLaMA-Factory 标准 DPO 格式（可同时用于 DPO 和 RM 训练）

rejected 类型设计（针对 Raymond 人设的 8 种失败模式）：
  1. too_formal      - 过于书面/正式，像官方客服
  2. too_long        - 回复过长，不像微信聊天
  3. no_catchphrase  - 完全没有 Raymond 口头禅
  4. emoji_heavy     - 大量堆砌 emoji
  5. self_qa         - 自问自答，不等对方回复
  6. overly_positive - 过于正能量，像心灵鸡汤/AI 助手
  7. wrong_language  - 文字太规整，没有网络用语/缩写
  8. melodramatic    - 情绪表达过于矫情/刻意

输出格式（LLaMA-Factory DPO 格式）：
[
  {
    "conversations": [  // 对话历史（不含最后一轮 gpt 回复）
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."},
      {"from": "human", "value": "..."}
    ],
    "chosen": {"from": "gpt", "value": "..."},   // 正确的 Raymond 风格回复
    "rejected": {"from": "gpt", "value": "..."}  // 故意错误的回复
  }
]
"""

import json
import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# =================== 路径配置 ===================
BASE_DIR = Path(__file__).parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
OUTPUT_DIR = Path(__file__).parent / "raw_generated"
OUTPUT_FILE = OUTPUT_DIR / "preference_data.json"

PERSONA_FILE = RESOURCES_DIR / "raymond_persona.json"
FEWSHOT_FILE = RESOURCES_DIR / "raymond_fewshot.json"

# =================== 生成配置 ===================
TARGET_TOTAL = 800          # 偏好对目标条数（DPO/RM 不需要像 SFT 那么多）
BATCH_SIZE = 3              # 每次 API 调用生成的偏好对数
DELAY_BETWEEN_CALLS = 1.0   # API 调用间隔（秒）

# =================== rejected 类型定义 ===================
# 每个类型：(描述, 权重, 详细说明给 Claude 的指令)
REJECTION_TYPES = {
    "too_formal": (
        "回复过于书面正式",
        15,
        "用标准书面语回复，句子完整，标点规范，语气正式，像官方客服或写作文，完全没有网络用语"
    ),
    "too_long": (
        "回复过于冗长",
        15,
        "回复超过 60 个字，说了一大段话，解释很多细节，像在写日记或发朋友圈，不像微信聊天"
    ),
    "no_catchphrase": (
        "没有任何 Raymond 口头禅",
        12,
        "回复虽然简短，但完全没用 66、哈、f、说白了、不好说、俺、无敌了、我真谢了等任何口头禅，"
        "显得太陌生/疏远，像一个普通人而不是 Raymond"
    ),
    "emoji_heavy": (
        "堆砌大量 emoji",
        10,
        "在回复中用 5 个以上 emoji，用各种表情图标来表达情绪，显得不像 Raymond 的真实聊天风格"
    ),
    "self_qa": (
        "自问自答不等回复",
        12,
        "在一条消息里先提问然后自己回答，或者说了很多话把对话堵死，不给对方留回复空间，"
        "像在写独白而不是聊天"
    ),
    "overly_positive": (
        "过于正能量/AI 助手风格",
        12,
        "回复充满正能量、鼓励和支持，像 AI 助手或心理咨询师，完全没有 Raymond 的毒舌、"
        "自嘲和随性，语气过于温和友好"
    ),
    "wrong_language": (
        "语言太规整没网络用语",
        12,
        "虽然回复简短，但用词太标准普通话，没有任何网络缩写、谐音字、粤语词、"
        "或 Raymond 特有的语言习惯，显得像教科书里的对话"
    ),
    "melodramatic": (
        "情绪表达矫情刻意",
        12,
        "把情绪表达得非常直白和戏剧化，用夸张的词语描述感受，像在写文艺小说，"
        "完全不符合 Raymond 把情绪藏在玩笑里、不轻易直接表达的特点"
    ),
}

# 预计算权重
REJECTION_KEYS = list(REJECTION_TYPES.keys())
REJECTION_WEIGHTS = [REJECTION_TYPES[k][1] for k in REJECTION_KEYS]

# 复用 SFT 数据的话题场景（保持一致性）
SCENARIOS = [
    ("两个好朋友的日常闲聊，话题随机，比如今天干了什么、吃了什么、有什么新鲜事", 15),
    ("深夜聊天，两人都睡不着，聊一些没营养但很舒服的话", 8),
    ("吐槽室友、邻居或者陌生人的奇葩行为，越无语越好", 10),
    ("吐槽美国生活的不便、文化差异，想念国内的东西", 10),
    ("吐槽作业多、考试难、教授讲得烂，学业压力", 8),
    ("叫朋友一起打游戏（铲），聊游戏里的事，炫耀战绩或者吐槽队友", 10),
    ("聊代码、bug、编程作业，用自嘲或骄傲的方式描述自己的技术", 8),
    ("讨论 AI 工具（GPT、Claude、Cursor 等），分享使用体验", 6),
    ("自己做了饭，结果翻车了或者出乎意料地好吃，聊做饭经过", 8),
    ("聊到感情话题，Raymond 从调侃慢慢变认真，说出真实的孤独感或想法", 7),
    ("安慰朋友，Raymond 先插科打诨然后认真说两句鼓励的话", 6),
    ("和樊明磊（米粒）的互动，互相损对方，叫宝宝，聊八卦", 10),
    ("朋友问 Raymond 要建议，Raymond 给出又毒又有道理的回答", 8),
]

SCENARIO_TEXTS = [s[0] for s in SCENARIOS]
SCENARIO_WEIGHTS_LIST = [s[1] for s in SCENARIOS]


def load_persona() -> dict:
    with open(PERSONA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fewshot() -> dict:
    with open(FEWSHOT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_teacher_system_prompt(persona: dict, fewshot: dict) -> str:
    """构建偏好数据生成的 system prompt"""
    p = persona
    sp = p["system_prompt"]

    prompt = f"""你是一个专门生成对话偏好训练数据的专家。你需要生成 Raymond 的 (chosen, rejected) 对话对，
用于训练一个对话 AI 的偏好对齐（DPO/RLHF）。

Raymond 的完整人设：
{sp}

补充细节：
- 身份：{p['basic_info']['身份']}，{p['basic_info']['所在地']}
- 日常：{p['basic_info']['日常']}

Raymond 的说话风格（非常重要）：
- 总体：{p['speaking_style']['总体特征']}
- 口头禅：{', '.join(p['speaking_style']['口头禅'][:8])}
- 句式习惯：{'; '.join(p['speaking_style']['句式习惯'][:4])}
- 回复长度：大多数 1-15 字，用 \\n 分隔多条短消息，绝对不写长段落

真实风格示例：
"""
    for example in fewshot["examples"][:6]:
        prompt += f"\n[{example['topic']}]\n"
        for turn in example["conversation"][:4]:
            role = "朋友" if turn["role"] == "human" else "Raymond"
            prompt += f"{role}: {turn['content']}\n"

    prompt += """
生成规则：
- chosen 必须体现 Raymond 的真实风格：短、碎、有口头禅、不做作
- rejected 必须明确体现指定的失败模式，但内容本身要合理（不能是乱说话）
- 对话历史要自然，为最后一轮 Raymond 的回复做好铺垫
- 不同样本的失败方式要尽量多样"""
    return prompt


def build_preference_request(scenario: str, rejection_key: str, batch_size: int) -> str:
    """构建单次偏好数据生成请求"""
    rejection_desc, _, rejection_instruction = REJECTION_TYPES[rejection_key]

    return f"""请生成 {batch_size} 个偏好训练样本。

话题场景：{scenario}
rejected 类型：{rejection_desc}
rejected 生成要求：{rejection_instruction}

输出格式（严格遵守）：
- "conversations"：对话历史，包含 system + 3-6 轮交替对话，最后一条必须是 "human"（朋友的消息）
- "chosen"：Raymond 用真实风格回复最后那条朋友消息
- "rejected"：用 "{rejection_desc}" 的方式回复同一条消息（内容逻辑要合理，只是风格/方式不对）

JSON 格式示例：
[
  {{
    "conversations": [
      {{"from": "system", "value": "你是Raymond...（简短系统提示）"}},
      {{"from": "human", "value": "你在干嘛"}},
      {{"from": "gpt", "value": "打游戏哈\\n摆烂中"}},
      {{"from": "human", "value": "作业不写吗"}}
    ],
    "chosen": {{"from": "gpt", "value": "明天再说哈\\n今天额度用完了"}},
    "rejected": {{"from": "gpt", "value": "我目前正在进行游戏娱乐活动，明天我会合理安排时间完成作业的，请你放心。"}}
  }}
]

只输出 JSON 数组，不要任何其他文字。"""


def parse_response(content: str) -> list:
    """解析 Claude 返回的 JSON，容错处理"""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return []


def validate_preference_sample(sample: dict) -> bool:
    """校验偏好样本格式"""
    if not isinstance(sample, dict):
        return False

    # 检查必需字段
    for field in ("conversations", "chosen", "rejected"):
        if field not in sample:
            return False

    convs = sample["conversations"]
    if not isinstance(convs, list) or len(convs) < 2:
        return False

    # 检查最后一条是 human（朋友消息）
    if convs[-1].get("from") != "human":
        return False

    # 检查 chosen 和 rejected
    for key in ("chosen", "rejected"):
        item = sample[key]
        if not isinstance(item, dict):
            return False
        if item.get("from") != "gpt":
            return False
        if not item.get("value", "").strip():
            return False

    # chosen 和 rejected 不应该完全一样
    if sample["chosen"]["value"] == sample["rejected"]["value"]:
        return False

    # rejected 应该和 chosen 有明显差异
    chosen_len = len(sample["chosen"]["value"])
    rejected_len = len(sample["rejected"]["value"])
    # 至少在长度上有 50% 差异，或内容明显不同
    # （粗略检查，详细清洗在 clean_preference_data.py 中做）
    return True


def load_existing(output_file: Path) -> list:
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"检测到已有数据：{len(data)} 条，继续追加生成")
        return data
    return []


def save_data(data: list, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 55)
    print("Raymond 偏好数据生成脚本（DPO / RLHF）")
    print("=" * 55)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 ANTHROPIC_API_KEY\n"
            "export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = Anthropic(api_key=api_key)

    print("加载 persona 和 fewshot 资源...")
    persona = load_persona()
    fewshot = load_fewshot()
    teacher_system = build_teacher_system_prompt(persona, fewshot)

    all_data = load_existing(OUTPUT_FILE)
    already_count = len(all_data)

    print(f"目标生成：{TARGET_TOTAL} 条偏好对")
    print(f"当前已有：{already_count} 条")
    print(f"还需生成：{max(0, TARGET_TOTAL - already_count)} 条")
    print()

    # 统计各 rejected 类型的已有数量（用于均衡采样）
    call_count = 0
    fail_count = 0

    while len(all_data) < TARGET_TOTAL:
        remaining = TARGET_TOTAL - len(all_data)
        batch = min(BATCH_SIZE, remaining)

        scenario = random.choices(SCENARIO_TEXTS, weights=SCENARIO_WEIGHTS_LIST, k=1)[0]
        rejection_key = random.choices(REJECTION_KEYS, weights=REJECTION_WEIGHTS, k=1)[0]
        rejection_desc = REJECTION_TYPES[rejection_key][0]

        print(
            f"[{len(all_data)}/{TARGET_TOTAL}] "
            f"场景: {scenario[:25]}... | "
            f"rejected: {rejection_desc}",
            end=" ", flush=True
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3000,
                system=teacher_system,
                messages=[
                    {"role": "user", "content": build_preference_request(scenario, rejection_key, batch)}
                ]
            )
            call_count += 1

            content = response.content[0].text
            samples = parse_response(content)

            valid_samples = [s for s in samples if validate_preference_sample(s)]
            all_data.extend(valid_samples)
            print(f"✓ 获得 {len(valid_samples)}/{len(samples)} 条")

            if call_count % 10 == 0:
                save_data(all_data, OUTPUT_FILE)
                print(f"  → 已保存 {len(all_data)} 条到文件")

            time.sleep(DELAY_BETWEEN_CALLS)

        except Exception as e:
            fail_count += 1
            print(f"✗ 失败: {e}")
            time.sleep(3)
            if fail_count > 10:
                print("连续失败次数过多，中止生成")
                break

    save_data(all_data, OUTPUT_FILE)

    print()
    print("=" * 55)
    print(f"生成完成！")
    print(f"总条数：{len(all_data)}")
    print(f"API 调用次数：{call_count}")
    print(f"失败次数：{fail_count}")
    print(f"输出文件：{OUTPUT_FILE}")
    print("下一步：运行 clean_preference_data.py 进行清洗")
    print("=" * 55)


if __name__ == "__main__":
    main()
