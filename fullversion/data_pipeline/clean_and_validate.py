"""
Step 2: 数据清洗与验证脚本

技术说明：
- 输入：raw_generated/distilled_data.json（蒸馏原始数据）
- 输出：processed/raymond_train.json（LLaMA-Factory ShareGPT 格式）
- 清洗策略：
  1. 格式校验：确保每条都有合法的 conversations 结构
  2. 长度过滤：过滤掉过短（无效）或过长（超显存）的对话
  3. 内容去重：基于对话哈希去掉完全重复的样本
  4. Raymond 回复质量检查：过滤掉 Raymond 说话风格严重偏离的样本
  5. 顺序打乱：shuffle 后输出，避免训练时场景扎堆
"""

import json
import random
import hashlib
from pathlib import Path

# =================== 路径配置 ===================
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "raw_generated" / "distilled_data.json"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "raymond_train.json"

# =================== 清洗参数 ===================
MIN_TURNS = 2          # human+gpt 最少轮数（太短没有学习价值）
MAX_TURNS = 20         # 最多轮数（太长超出训练 cutoff）
MIN_GPT_CHARS = 2      # Raymond 单条回复最少字符数
MAX_GPT_CHARS = 500    # Raymond 单条回复最多字符数（过长说明风格跑偏）
MAX_TOTAL_CHARS = 2000 # 整条对话总字符数上限（对应 cutoff_len=2048）

# 风格质量检查：Raymond 至少要有这些特征之一才算合格
STYLE_MARKERS = [
    "哈", "66", "ff", "f\n", "说白了", "不好说", "俺", "无敌了",
    "我真谢了", "我靠", "卧槽", "铲", "宝宝", "米粒", "6\n",
    "？\n", "\n", "疑似", "恭喜"
]

RANDOM_SEED = 42


def load_raw(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_hash(sample: dict) -> str:
    """基于对话内容生成哈希，用于去重"""
    convs = sample.get("conversations", [])
    text = "||".join(
        f"{c['from']}:{c['value']}"
        for c in convs
        if c["from"] in ("human", "gpt")
    )
    return hashlib.md5(text.encode()).hexdigest()


def check_format(sample: dict) -> tuple[bool, str]:
    """格式校验"""
    convs = sample.get("conversations")
    if not isinstance(convs, list) or len(convs) < 1:
        return False, "无 conversations"

    has_human = any(c.get("from") == "human" for c in convs)
    has_gpt = any(c.get("from") == "gpt" for c in convs)
    if not has_human or not has_gpt:
        return False, "缺少 human 或 gpt 消息"

    for c in convs:
        if c.get("from") not in ("system", "human", "gpt"):
            return False, f"非法 from 值: {c.get('from')}"
        if not isinstance(c.get("value", ""), str):
            return False, "value 不是字符串"

    return True, "ok"


def check_length(sample: dict) -> tuple[bool, str]:
    """长度过滤"""
    convs = sample.get("conversations", [])
    dialogue = [c for c in convs if c["from"] in ("human", "gpt")]

    if len(dialogue) < MIN_TURNS:
        return False, f"轮数过少: {len(dialogue)}"
    if len(dialogue) > MAX_TURNS:
        return False, f"轮数过多: {len(dialogue)}"

    total_chars = sum(len(c.get("value", "")) for c in dialogue)
    if total_chars > MAX_TOTAL_CHARS:
        return False, f"总字符过多: {total_chars}"

    return True, "ok"


def check_gpt_quality(sample: dict) -> tuple[bool, str]:
    """Raymond 回复质量检查"""
    convs = sample.get("conversations", [])
    gpt_msgs = [c for c in convs if c["from"] == "gpt"]

    for msg in gpt_msgs:
        val = msg.get("value", "")
        # 单条回复不能过短或过长
        if len(val) < MIN_GPT_CHARS:
            return False, f"Raymond 回复过短: '{val}'"
        if len(val) > MAX_GPT_CHARS:
            return False, f"Raymond 回复过长: {len(val)} 字符"

    # 合并所有 Raymond 回复检查风格标记
    all_gpt_text = " ".join(c.get("value", "") for c in gpt_msgs)
    has_style = any(marker in all_gpt_text for marker in STYLE_MARKERS)
    if not has_style:
        return False, "缺少 Raymond 风格标记"

    return True, "ok"


def clean_value(text: str) -> str:
    """清理文本中的异常字符"""
    # 去掉首尾多余空格，保留内部换行
    return text.strip()


def clean_sample(sample: dict) -> dict:
    """清理单条样本的文本内容"""
    new_convs = []
    for c in sample.get("conversations", []):
        new_c = {
            "from": c["from"],
            "value": clean_value(c.get("value", ""))
        }
        if new_c["value"]:  # 去掉空消息
            new_convs.append(new_c)
    return {"conversations": new_convs}


def main():
    print("=" * 50)
    print("Step 2: 数据清洗与验证")
    print("=" * 50)

    # 加载原始数据
    print(f"\n加载原始数据: {INPUT_FILE}")
    raw_data = load_raw(INPUT_FILE)
    print(f"原始条数: {len(raw_data)}")

    # 统计各步骤过滤情况
    stats = {
        "格式不合法": 0,
        "长度不合规": 0,
        "质量不达标": 0,
        "重复数据": 0,
        "通过": 0,
    }

    seen_hashes = set()
    cleaned = []

    for i, sample in enumerate(raw_data):
        # 1. 格式校验
        ok, reason = check_format(sample)
        if not ok:
            stats["格式不合法"] += 1
            continue

        # 2. 长度过滤
        ok, reason = check_length(sample)
        if not ok:
            stats["长度不合规"] += 1
            continue

        # 3. 质量检查
        ok, reason = check_gpt_quality(sample)
        if not ok:
            stats["质量不达标"] += 1
            continue

        # 4. 去重
        h = get_hash(sample)
        if h in seen_hashes:
            stats["重复数据"] += 1
            continue
        seen_hashes.add(h)

        # 5. 清理文本
        cleaned_sample = clean_sample(sample)
        cleaned.append(cleaned_sample)
        stats["通过"] += 1

    # 打乱顺序
    random.seed(RANDOM_SEED)
    random.shuffle(cleaned)

    # 保存
    save_output(cleaned, OUTPUT_FILE)

    # 输出报告
    print("\n=== 清洗报告 ===")
    for k, v in stats.items():
        pct = v / len(raw_data) * 100
        print(f"  {k}: {v} 条 ({pct:.1f}%)")
    print(f"\n最终输出: {len(cleaned)} 条")
    print(f"保存至: {OUTPUT_FILE}")

    # 抽样展示
    print("\n=== 清洗后样本抽检（3条）===")
    for sample in random.sample(cleaned, min(3, len(cleaned))):
        print()
        for c in sample["conversations"]:
            if c["from"] == "system":
                print(f"[system]: {c['value'][:50]}...")
                continue
            role = "朋友" if c["from"] == "human" else "Raymond"
            print(f"{role}: {c['value']}")
        print("---")

    # 统计对话轮数分布
    turn_counts = [
        len([c for c in s["conversations"] if c["from"] in ("human", "gpt")])
        for s in cleaned
    ]
    print(f"\n=== 轮数分布 ===")
    print(f"  平均: {sum(turn_counts)/len(turn_counts):.1f} 轮")
    print(f"  最少: {min(turn_counts)} 轮")
    print(f"  最多: {max(turn_counts)} 轮")

    print("\n✅ Step 2 完成！可以进入 Step 3 训练了。")


if __name__ == "__main__":
    main()
