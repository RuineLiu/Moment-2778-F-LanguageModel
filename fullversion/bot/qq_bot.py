"""
Raymond QQ 机器人
================

技术栈：
- 框架：NoneBot2 (https://nonebot.dev)
- 适配器：OneBot V11（兼容 LLOneBot / go-cqhttp / Lagrange 等）
- 推理：raymond_core.py（本地 Ollama）

部署前提：
1. 安装 LLOneBot（推荐）或 go-cqhttp：
   - LLOneBot：https://llonebot.github.io/zh-CN/guide/getting-started
   - 配置 HTTP 正向连接到 localhost:8080，或 WebSocket 反向连接到 localhost:8080/onebot/v11/ws
2. pip install nonebot2 nonebot-adapter-onebot

启动：
   python qq_bot.py

会话管理：
- 私聊：直接和 Raymond 聊，保留最近 10 轮上下文
- 群聊：@机器人 才触发（防止刷屏），保留最近 5 轮上下文
- 重置命令：发 "重置" 或 "/reset" 清空上下文
"""

import sys
from pathlib import Path

# 确保能 import raymond_core
sys.path.insert(0, str(Path(__file__).parent))

from raymond_core import chat, check_ollama_status

# =================== NoneBot2 ===================
import nonebot
from nonebot.adapters.onebot.v11 import (
    Adapter as OneBot11Adapter,
    Bot,
    Event,
    Message,
    MessageEvent,
    PrivateMessageEvent,
    GroupMessageEvent,
)
from nonebot import on_message, on_command
from nonebot.rule import to_me
from nonebot.typing import T_State

# =================== 上下文管理 ===================
# key: user_id (私聊) 或 f"{group_id}_{user_id}" (群聊)
# value: list of {"role": "user/assistant", "content": "..."}
_history: dict[str, list] = {}

MAX_PRIVATE_TURNS = 10   # 私聊保留最近 10 轮
MAX_GROUP_TURNS = 5      # 群聊保留最近 5 轮


def get_key(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"{event.group_id}_{event.user_id}"
    return str(event.user_id)


def get_history(key: str) -> list:
    return _history.get(key, [])


def update_history(key: str, user_msg: str, bot_reply: str, max_turns: int):
    h = _history.setdefault(key, [])
    h.append({"role": "user", "content": user_msg})
    h.append({"role": "assistant", "content": bot_reply})
    # 保留最近 max_turns 轮（每轮 2 条）
    if len(h) > max_turns * 2:
        _history[key] = h[-(max_turns * 2):]


def clear_history(key: str):
    _history.pop(key, None)


# =================== Handler 定义 ===================

# 1. 私聊：所有消息都触发
private_chat = on_message(
    rule=lambda event: isinstance(event, PrivateMessageEvent),
    priority=10,
    block=True,
)

# 2. 群聊：@机器人 才触发
group_chat = on_message(
    rule=to_me() & (lambda event: isinstance(event, GroupMessageEvent)),
    priority=10,
    block=True,
)

# 3. 重置命令（私聊+群聊）
reset_cmd = on_command("reset", aliases={"重置", "清空"}, priority=5, block=True)


@private_chat.handle()
async def handle_private(bot: Bot, event: PrivateMessageEvent):
    user_input = event.get_plaintext().strip()
    if not user_input:
        return

    key = get_key(event)
    history = get_history(key)

    reply = chat(user_input, history)
    update_history(key, user_input, reply, MAX_PRIVATE_TURNS)

    await private_chat.finish(Message(reply))


@group_chat.handle()
async def handle_group(bot: Bot, event: GroupMessageEvent):
    # 去掉 @机器人 部分，只保留纯文本
    user_input = event.get_plaintext().strip()
    if not user_input:
        return

    key = get_key(event)
    history = get_history(key)

    reply = chat(user_input, history)
    update_history(key, user_input, reply, MAX_GROUP_TURNS)

    # 群聊回复时 @ 用户
    from nonebot.adapters.onebot.v11 import MessageSegment
    msg = MessageSegment.at(event.user_id) + " " + reply
    await group_chat.finish(Message(msg))


@reset_cmd.handle()
async def handle_reset(bot: Bot, event: MessageEvent):
    key = get_key(event)
    clear_history(key)
    await reset_cmd.finish("哈 忘了")


# =================== 启动配置 ===================
def main():
    if not check_ollama_status():
        print("❌ Ollama 未启动！请先运行: ollama serve")
        sys.exit(1)

    print("✅ Ollama 正常，Raymond 准备就绪")

    nonebot.init(
        # NoneBot2 基础配置
        host="0.0.0.0",
        port=8080,
        # OneBot V11 配置
        onebot_access_token="",   # 如果 LLOneBot 设置了 token，填在这里
    )

    driver = nonebot.get_driver()
    driver.register_adapter(OneBot11Adapter)

    nonebot.load_builtin_plugins("echo")  # 可选：内置 echo 插件

    print("\n=== Raymond QQ Bot 启动 ===")
    print("监听地址: http://0.0.0.0:8080")
    print("请确保 LLOneBot/go-cqhttp 已配置反向 WebSocket 连接到此地址")
    print("私聊：直接发消息")
    print("群聊：@机器人 触发")
    print("重置：发送 '重置' 或 '/reset'")
    print("===========================\n")

    nonebot.run()


if __name__ == "__main__":
    main()
