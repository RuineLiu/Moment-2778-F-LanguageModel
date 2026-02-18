"""
Raymond Agent - 命令行聊天入口
"""
import os
from dotenv import load_dotenv

# 在所有其他导入之前加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import create_agent
from agent.memory import LongTermMemory
from agent.config import LLM_PROVIDER, get_model_name, FAISS_INDEX_PATH


def main():
    print("=" * 40)
    print("  Raymond Agent")
    print(f"  模型: {LLM_PROVIDER} / {get_model_name()}")
    print("=" * 40)

    print("\n正在初始化...")

    # 初始化 agent
    agent = create_agent()

    # 显示记忆状态
    try:
        mem = LongTermMemory(FAISS_INDEX_PATH)
        print(f"长期记忆已加载: {mem.get_all_count()} 条")
    except Exception as e:
        print(f"长期记忆加载失败: {e}")

    print("\n命令:")
    print("  /memory  - 查看长期记忆统计")
    print("  /search <关键词> - 搜索长期记忆")
    print("  /clear   - 清空当前对话历史")
    print("  exit     - 退出")
    print()

    # 对话状态
    state = {
        "messages": [],
        "retrieved_memories": [],
        "should_extract_memory": False,
    }

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见哈")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("66拜拜")
            break

        # 处理命令
        if user_input.startswith("/"):
            if user_input.strip().lower() == "/clear":
                state = {
                    "messages": [],
                    "retrieved_memories": [],
                    "should_extract_memory": False,
                }
                print("对话历史已清空\n")
                continue
            handle_command(user_input)
            continue

        # 添加用户消息到状态
        state["messages"].append(HumanMessage(content=user_input))

        # 运行 agent
        try:
            result = agent.invoke(state)
            state = result

            # 获取最新的 AI 回复
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(f"Raymond: {last_msg.content}")
            print()

        except Exception as e:
            print(f"\n[Error] {e}")
            print("检查: API key 是否设置？Ollama 是否运行？\n")


def handle_command(cmd: str):
    """处理 / 开头的命令"""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()

    if command == "/memory":
        try:
            mem = LongTermMemory(FAISS_INDEX_PATH)
            print(f"长期记忆总数: {mem.get_all_count()} 条")
        except Exception as e:
            print(f"读取记忆失败: {e}")

    elif command == "/search":
        if len(parts) < 2:
            print("用法: /search <关键词>")
            return
        query = parts[1]
        try:
            mem = LongTermMemory(FAISS_INDEX_PATH)
            results = mem.search(query, top_k=5)
            if results:
                print(f"找到 {len(results)} 条相关记忆:")
                for i, r in enumerate(results, 1):
                    print(f"  {i}. {r}")
            else:
                print("没有找到相关记忆")
        except Exception as e:
            print(f"搜索失败: {e}")

    elif command == "/clear":
        print("对话历史已清空")

    else:
        print(f"未知命令: {command}")

    print()


if __name__ == "__main__":
    main()
