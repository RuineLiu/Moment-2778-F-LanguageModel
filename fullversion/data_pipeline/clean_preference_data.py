"""
Step 2b: 偏好数据清洗与验证脚本
对 generate_preference_data.py 生成的原始偏好对进行清洗

清洗步骤：
1. 格式校验：chosen/rejected 字段完整
2. 长度检查：chosen 不超过 300 字，rejected 不超过 500 字
3. 差异度检查：chosen 和 rejected 必须有足够差异（Jaccard 距离）
4. 质量检查：chosen 应包含 Raymond 风格标记
5. 去重：MD5 哈希去除重复样本
6. 打乱：随机顺序
"""

import json
import hashlib
import random
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "raw_generated" / "preference_data.json"
OUTPUT_FILE = Path(__file__).parent / "processed" / "raymond_preference.json"

# Raymond 风格标记（chosen 中至少要有其中之一）
STYLE_MARKERS = [
    "66", "哈", "f", "说白了", "不好说", "俺", "无敌了", "我真谢了", "我靠",
    "铲", "宝宝", "破", "寄", "nb", "gg", "\n",  # \n 代表多条短消息风格
]

# 长度阈值
MAX_CHOSEN_LEN = 300
MAX_REJECTED_LEN = 500
MIN_CHOSEN_LEN = 2
MAX_CONVERSATIONS_TURNS = 16  # conversations 最多多少条消息


def jaccard_similarity(s1: str, s2: str) -> float:
    """用字符级 bigram 计算 Jaccard 相似度"""
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))
    bg1 = bigrams(s1)
    bg2 = bigrams(s2)
    if not bg1 and not bg2:
        return 1.0
    intersection = len(bg1 & bg2)
    union = len(bg1 | bg2)
    return intersection / union if union else 0.0


def check_format(sample: dict) -> tuple[bool, str]:
    """格式校验"""
    for field in ("conversations", "chosen", "rejected"):
        if field not in sample:
            return False, f"缺少字段: {field}"

    convs = sample["conversations"]
    if not isinstance(convs, list) or len(convs) < 2:
        return False, "conversations 太短"

    if len(convs) > MAX_CONVERSATIONS_TURNS:
        return False, f"conversations 太长: {len(convs)} 轮"

    if convs[-1].get("from") != "human":
        return False, "conversations 末尾不是 human"

    for key in ("chosen", "rejected"):
        item = sample[key]
        if not isinstance(item, dict) or item.get("from") != "gpt":
            return False, f"{key} 格式错误"
        if not item.get("value", "").strip():
            return False, f"{key} 内容为空"

    return True, ""


def check_length(sample: dict) -> tuple[bool, str]:
    chosen_len = len(sample["chosen"]["value"])
    rejected_len = len(sample["rejected"]["value"])

    if chosen_len < MIN_CHOSEN_LEN:
        return False, f"chosen 过短: {chosen_len}"
    if chosen_len > MAX_CHOSEN_LEN:
        return False, f"chosen 过长: {chosen_len}"
    if rejected_len > MAX_REJECTED_LEN:
        return False, f"rejected 过长: {rejected_len}"

    return True, ""


def check_diversity(sample: dict) -> tuple[bool, str]:
    """chosen 和 rejected 必须有足够差异"""
    chosen = sample["chosen"]["value"]
    rejected = sample["rejected"]["value"]

    sim = jaccard_similarity(chosen, rejected)
    if sim > 0.7:
        return False, f"chosen/rejected 过于相似: sim={sim:.2f}"

    # 额外检查：不能完全一样
    if chosen.strip() == rejected.strip():
        return False, "chosen 和 rejected 完全相同"

    return True, ""


def check_style_markers(sample: dict) -> tuple[bool, str]:
    """chosen 应包含 Raymond 风格标记（至少一个）"""
    chosen = sample["chosen"]["value"].lower()
    has_marker = any(marker.lower() in chosen for marker in STYLE_MARKERS)
    if not has_marker:
        return False, "chosen 缺少 Raymond 风格标记"
    return True, ""


def compute_hash(sample: dict) -> str:
    """基于 conversations + chosen + rejected 计算 MD5"""
    key = json.dumps({
        "c": sample["conversations"],
        "chosen": sample["chosen"]["value"],
        "rejected": sample["rejected"]["value"],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def main():
    print("=" * 55)
    print("偏好数据清洗脚本")
    print("=" * 55)

    if not INPUT_FILE.exists():
        print(f"找不到输入文件: {INPUT_FILE}")
        print("请先运行 generate_preference_data.py")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"原始数据：{len(raw_data)} 条")

    stats = {
        "format_fail": 0,
        "length_fail": 0,
        "diversity_fail": 0,
        "style_fail": 0,
        "duplicate": 0,
        "passed": 0,
    }

    seen_hashes = set()
    cleaned = []

    checks = [
        ("format_fail",    check_format),
        ("length_fail",    check_length),
        ("diversity_fail", check_diversity),
        ("style_fail",     check_style_markers),
    ]

    for sample in raw_data:
        failed = False
        for stat_key, check_fn in checks:
            ok, reason = check_fn(sample)
            if not ok:
                stats[stat_key] += 1
                failed = True
                break

        if failed:
            continue

        h = compute_hash(sample)
        if h in seen_hashes:
            stats["duplicate"] += 1
            continue

        seen_hashes.add(h)
        cleaned.append(sample)
        stats["passed"] += 1

    # 打乱顺序
    random.seed(42)
    random.shuffle(cleaned)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print()
    print(f"格式校验失败：{stats['format_fail']}")
    print(f"长度检查失败：{stats['length_fail']}")
    print(f"差异度不足：  {stats['diversity_fail']}")
    print(f"风格标记缺失：{stats['style_fail']}")
    print(f"重复去除：    {stats['duplicate']}")
    print(f"通过（保留）：{stats['passed']}")
    pass_rate = stats["passed"] / len(raw_data) * 100 if raw_data else 0
    print(f"通过率：      {pass_rate:.1f}%")
    print()
    print(f"输出文件：{OUTPUT_FILE}")
    print("下一步：")
    print("  - DPO 路径：上传到 Colab，运行 dpo_train.ipynb")
    print("  - RLHF 路径：上传到 Colab，先运行 reward_model_train.ipynb，再运行 rlhf_ppo_train.ipynb")
    print("=" * 55)


if __name__ == "__main__":
    main()
