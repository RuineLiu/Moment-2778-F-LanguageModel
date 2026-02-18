# Raymond Bot 部署指南

## 前提条件

**Ollama 必须在后台运行：**
```bash
ollama serve          # 启动 Ollama 服务
ollama list           # 确认 raymond 模型已加载
```

---

## QQ 机器人部署

### 1. 安装 NoneBot2

```bash
pip install nonebot2 nonebot-adapter-onebot
```

### 2. 选择 QQ 协议端（二选一）

**推荐：LLOneBot**（基于 NTQQ，不需要旧版 QQ）
- 下载：https://llonebot.github.io/zh-CN/guide/getting-started
- 在 NTQQ 插件里开启
- 配置：**正向 HTTP** 或 **反向 WebSocket**，连接到 `http://localhost:8080`

**备选：Lagrange.OneBot**（协议实现，无需安装 QQ 客户端）
- GitHub: https://github.com/LagrangeDev/Lagrange.Core
- 扫码登录后，配置 WebSocket 连接到 `localhost:8080`

### 3. 启动 Raymond QQ Bot

```bash
cd /Users/jorahmormont/PycharmProjects/Moment-2778-F-LanguageModel/fullversion
python bot/qq_bot.py
```

### 4. 使用说明

| 场景 | 操作 |
|------|------|
| 私聊 | 直接发消息 |
| 群聊 | @机器人 + 消息 |
| 重置上下文 | 发 `重置` 或 `/reset` |

---

## 微信机器人部署

### 方案 A：WeChatFerry（推荐）

**需要：Windows 环境（或 Windows 虚拟机）**

WeChatFerry 通过 hook 微信 Windows 客户端实现，是目前最稳定的微信 Bot 方案。

```bash
# Windows 上安装
pip install wcferry

# 打开微信（需要 3.9.10.x 版本）并登录

# 启动 bot
python wechat_bot.py
# 选 1（WeChatFerry）
```

微信版本下载：https://github.com/lich0821/WeChatFerry/releases（含兼容的微信版本）

### 方案 B：itchat（Mac 可直接用，但可能不可用）

itchat 使用微信网页版协议。微信从 2023 年起大范围限制网页版登录，**新注册的账号大概率无法使用**。

```bash
pip install itchat-uos

python wechat_bot.py
# 选 2（itchat）
# 扫码登录
```

### 方案 C：Mac 上的 WeChatFerry（使用 Wine）

暂不推荐，配置复杂且不稳定。建议用 Windows 虚拟机（Parallels/VMware）。

---

## 架构图

```
[QQ/微信消息]
      ↓
[qq_bot.py / wechat_bot.py]  ← 收发消息、管理上下文
      ↓
[raymond_core.py]            ← 推理封装
      ↓
[Ollama /api/generate]       ← 本地推理（raymond 模型）
      ↓
[raymond-q4_k_m.gguf]        ← 量化后的 Qwen3-4B 模型
```

---

## 注意事项

- Ollama 默认监听 `localhost:11434`，Bot 和 Ollama 需在同一台机器
- 微信/QQ 机器人存在封号风险，建议用小号
- 群聊中机器人被 @ 才回复，避免刷屏
