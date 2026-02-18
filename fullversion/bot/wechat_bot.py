"""
Raymond 微信机器人
================

技术栈：
- 框架：WeChatFerry (wcf) - 基于协议的微信 Hook 方案
  GitHub: https://github.com/lich0821/WeChatFerry
- 推理：raymond_core.py（本地 Ollama）

⚠️  重要说明：
微信没有官方 Bot API，所有方案都是 hook 微信客户端。
WeChatFerry 是目前最稳定的方案，支持：
- Windows 上 hook 微信客户端（需要 Windows 或 Wine/虚拟机）
- 接收私聊、群聊消息
- 发送文本消息

如果你在 Mac 上运行，有两种选择：
A) Windows 虚拟机（推荐）：在虚拟机里运行 wcf + 本脚本
B) 使用 itchat（微信网页版）：功能更有限，稳定性差，见下方备注

安装：
   pip install wcferry

部署：
1. 在 Windows 上打开微信 (3.9.10.x 版本)
2. 启动 wcf：python -m wcferry (后台运行)
3. 运行本脚本：python wechat_bot.py

会话管理：
- 私聊：直接回复，保留最近 10 轮上下文
- 群聊：@机器人 才触发，保留最近 5 轮上下文
- 重置：发 "重置" 清空上下文
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from raymond_core import chat, check_ollama_status

# =================== 上下文管理（与 qq_bot.py 共享逻辑） ===================
_history: dict[str, list] = {}
MAX_PRIVATE_TURNS = 10
MAX_GROUP_TURNS = 5


def get_history(key: str) -> list:
    return _history.get(key, [])


def update_history(key: str, user_msg: str, bot_reply: str, max_turns: int):
    h = _history.setdefault(key, [])
    h.append({"role": "user", "content": user_msg})
    h.append({"role": "assistant", "content": bot_reply})
    if len(h) > max_turns * 2:
        _history[key] = h[-(max_turns * 2):]


def clear_history(key: str):
    _history.pop(key, None)


# =================== WeChatFerry 版本 ===================
def run_wcf_bot():
    """
    使用 WeChatFerry 运行微信机器人
    需要 Windows 环境 + 微信客户端
    """
    try:
        from wcferry import Wcf, WxMsg
    except ImportError:
        print("❌ 未安装 wcferry，请运行: pip install wcferry")
        print("   注意：wcferry 需要 Windows 环境")
        return

    wcf = Wcf()

    if not wcf.is_login():
        print("❌ 微信未登录，请先打开微信并登录")
        wcf.cleanup()
        return

    self_wxid = wcf.get_self_wxid()
    print(f"✅ 微信已登录，wxid: {self_wxid}")
    print("Raymond 微信机器人启动中...\n")

    def handle_message(msg: WxMsg):
        """处理收到的消息"""
        # 只处理文本消息
        if msg.type != 1:
            return

        # 跳过自己发的消息
        if msg.sender == self_wxid:
            return

        content = msg.content.strip()
        if not content:
            return

        is_group = msg.from_group()

        if is_group:
            # 群聊：只处理 @机器人 的消息
            # wcf 格式：@Raymond\u2005消息内容
            if self_wxid not in msg.xml:
                return

            # 去掉 @ 提及部分
            import re
            content = re.sub(r"@\S+\s*", "", content).strip()
            if not content:
                return

            key = f"{msg.roomid}_{msg.sender}"
            history = get_history(key)

            if content in ("重置", "/reset"):
                clear_history(key)
                wcf.send_text("哈 忘了", msg.roomid, msg.sender)
                return

            reply = chat(content, history)
            update_history(key, content, reply, MAX_GROUP_TURNS)

            # 群聊回复 @发送者
            wcf.send_text(reply, msg.roomid, msg.sender)

        else:
            # 私聊
            key = msg.sender
            history = get_history(key)

            if content in ("重置", "/reset"):
                clear_history(key)
                wcf.send_text("哈 忘了", msg.sender)
                return

            reply = chat(content, history)
            update_history(key, content, reply, MAX_PRIVATE_TURNS)

            wcf.send_text(reply, msg.sender)

    # 启动消息接收
    wcf.enable_receiving_msg()

    print("=== Raymond 微信 Bot 运行中 ===")
    print("私聊：直接发消息")
    print("群聊：@机器人 触发")
    print("重置：发送 '重置'")
    print("Ctrl+C 退出")
    print("================================\n")

    try:
        while True:
            msg = wcf.get_msg()
            if msg:
                # 在新线程处理，避免阻塞接收
                threading.Thread(
                    target=handle_message, args=(msg,), daemon=True
                ).start()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n正在退出...")
    finally:
        wcf.cleanup()


# =================== itchat 备用版本（Mac 可用，但不稳定） ===================
def run_itchat_bot():
    """
    使用 itchat 运行微信机器人（微信网页版协议）
    - 优点：纯 Python，跨平台，无需 Windows
    - 缺点：微信已限制网页版登录，新号大多无法使用；功能受限
    需要: pip install itchat-uos
    """
    try:
        import itchat
        from itchat.content import TEXT
    except ImportError:
        print("❌ 未安装 itchat-uos，请运行: pip install itchat-uos")
        return

    @itchat.msg_register(TEXT)
    def handle_private(msg):
        """处理私聊消息"""
        content = msg.text.strip()
        if not content:
            return

        key = msg.fromUserName
        history = get_history(key)

        if content in ("重置", "/reset"):
            clear_history(key)
            return "哈 忘了"

        reply = chat(content, history)
        update_history(key, content, reply, MAX_PRIVATE_TURNS)
        return reply

    @itchat.msg_register(TEXT, isGroupChat=True)
    def handle_group(msg):
        """处理群消息（只响应@机器人）"""
        if not msg.isAt:
            return

        content = msg.text.strip()
        # 去掉 @ 提及
        import re
        content = re.sub(r"@\S+\s*", "", content).strip()
        if not content:
            return

        key = f"{msg.fromUserName}_{msg.actualNickName}"
        history = get_history(key)

        if content in ("重置", "/reset"):
            clear_history(key)
            itchat.send("哈 忘了", msg.fromUserName)
            return

        reply = chat(content, history)
        update_history(key, content, reply, MAX_GROUP_TURNS)

        # 群聊回复时 @发送者
        itchat.send(f"@{msg.actualNickName}\u2005{reply}", msg.fromUserName)

    print("=== 使用 itchat（微信网页版）===")
    print("⚠️  注意：itchat 依赖微信网页版，许多账号已无法使用")
    print("扫码登录中...\n")

    itchat.auto_login(hotReload=True)  # hotReload=True 保存登录状态，避免重复扫码

    print("\n=== Raymond 微信 Bot 运行中 ===")
    print("私聊：直接发消息")
    print("群聊：@机器人 触发")
    print("重置：发送 '重置'")
    print("================================\n")

    itchat.run()


# =================== 主入口 ===================
if __name__ == "__main__":
    if not check_ollama_status():
        print("❌ Ollama 未启动！请先运行: ollama serve")
        sys.exit(1)

    print("✅ Ollama 正常，Raymond 准备就绪\n")

    print("选择微信机器人方案：")
    print("  1. WeChatFerry（推荐，需 Windows 环境）")
    print("  2. itchat（跨平台，网页版协议，可能不可用）")
    print()

    choice = input("请选择 [1/2]: ").strip()

    if choice == "2":
        run_itchat_bot()
    else:
        run_wcf_bot()
