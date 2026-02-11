import json
import os
import re


class ChatDataCleaner:
    def __init__(self, raw_data_file, output_file, user_config_file, time_threshold_sec=300):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.time_threshold_sec = time_threshold_sec
        self.user_config_file = user_config_file

    def load_user_config(self):
        config = {}
        try:
            if not os.path.exists(self.user_config_file):
                print(f"警告: 配置文件 {self.user_config_file} 未找到。")
                return config
            with open(self.user_config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        config[key.strip()] = value.strip()
        except Exception as e:
            print(f"加载配置出错: {e}")
        return config

    def normalize_content(self, content):
        """
        标准化内容
        """
        if not content:
            return ""

        pattern = r'[\u2000-\u200B\u3000\u00A0\t]+'
        # 将所有特殊空格替换为标准空格 ' '
        content = re.sub(pattern, ' ', content)
        # 将连续的多个标准空格合并为一个
        content = re.sub(r' {2,}', ' ', content)

        return content.strip()

    def is_valid_text_message(self, content):
        if not content: return False

        # 过滤 XML、CDATA
        if "<?xml" in content or "CDATA" in content or "<msg" in content:
            return False

        # 过滤占位符
        invalid_markers = [
            "[图片]", "[语音]", "[视频]", "[表情]", "[动画表情]",
            "[文件]", "[小程序]", "[位置]", "[转账]", "[合并转发]",
            "[聊天记录]", "[通话]"
        ]

        for marker in invalid_markers:
            if marker in content:
                return False

        # 过滤URL
        if content.startswith("http://") or content.startswith("https://"):
            return False

        return True

    def clean_and_format_chat_data(self):
        config = self.load_user_config()
        TARGET_USER = config.get("TARGET_USER")
        TARGET_ASSISTANT = config.get("TARGET_ASSISTANT")

        if not TARGET_USER or not TARGET_ASSISTANT:
            print("错误: 缺少用户配置。")
            return

        print(f"开始清洗... User: {TARGET_USER} <-> Assistant: {TARGET_ASSISTANT}")

        with open(self.raw_data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        conversations = []
        current_session = []
        last_timestamp = 0

        dropped_count = 0

        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if item.get("_type") != "message":
                continue

            sender = item.get("accountName")
            raw_content = item.get("content", "")
            timestamp = item.get("timestamp", 0)

            # 过滤非目标用户/助手的消息
            if sender not in [TARGET_USER, TARGET_ASSISTANT]:
                continue

            content = self.normalize_content(raw_content)

            # 过滤无效消息
            if not self.is_valid_text_message(content):
                dropped_count += 1
                continue

            role = "user" if sender == TARGET_USER else "assistant"

            # 会话分割逻辑
            if last_timestamp != 0 and (timestamp - last_timestamp > self.time_threshold_sec):
                if current_session:
                    conversations.append(current_session)
                current_session = []

            current_session.append({
                "role": role,
                "content": content,
                "timestamp": timestamp
            })
            last_timestamp = timestamp

        if current_session:
            conversations.append(current_session)

        finetune_data = []

        # 定义 System Prompt
        SYSTEM_PROMPT = "You are Raymond, a humorous chatter with a unique sarcastic style."

        for session in conversations:
            if not session: continue

            # 合并连续相同角色的消息
            merged_messages = []
            for msg in session:
                if not merged_messages:
                    merged_messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    last_msg = merged_messages[-1]
                    if last_msg["role"] == msg["role"]:
                        last_msg["content"] += "\n" + msg["content"]
                    else:
                        merged_messages.append({"role": msg["role"], "content": msg["content"]})

            # 如果第一条是assistant，删除
            while merged_messages and merged_messages[0]["role"] == "assistant":
                merged_messages.pop(0)

            # 如果最后一条是user，删除
            while merged_messages and merged_messages[-1]["role"] == "user":
                merged_messages.pop()

            # 至少需要一组对话
            if len(merged_messages) < 2:
                continue


            sharegpt_conversations = []
            for msg in merged_messages:
                # 映射角色
                from_role = "human" if msg["role"] == "user" else "gpt"
                sharegpt_conversations.append({
                    "from": from_role,
                    "value": msg["content"]
                })

            entry = {
                "system": SYSTEM_PROMPT,
                "conversations": sharegpt_conversations
            }
            finetune_data.append(entry)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(finetune_data, f, indent=2, ensure_ascii=False)

        print(f"清洗完成！")
        print(f"- 丢弃无效消息: {dropped_count} 条")
        print(f"- 生成有效对话片段: {len(finetune_data)} 个")
        print(f"- 结果已保存至: {self.output_file}")


if __name__ == "__main__":
    raw_data_file = "GroupChat_History.jsonl"
    output_filename = "raymond_sharegpt.json"
    user_config_file = "../resources/user_config.properties"

    if os.path.exists(raw_data_file):
        cleaner = ChatDataCleaner(raw_data_file, output_filename, user_config_file)
        cleaner.clean_and_format_chat_data()
    else:
        print(f"未找到文件: {raw_data_file}")