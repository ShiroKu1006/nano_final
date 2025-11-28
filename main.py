import os
import asyncio
import json

import discord
from discord.ext import commands
from aiohttp import web
from dotenv import load_dotenv
load_dotenv()

# ========== 基本設定 ==========
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")  # 建議用環境變數放 token
GUILD_ID = int(os.getenv("GUILD_ID") )       # 改成你的伺服器 ID（可選）
CHANNEL_ID = int(os.getenv("CHANNEL_ID")  )         # 改成你要發訊息的文字頻道 ID
print(DISCORD_TOKEN)
intents = discord.Intents.default()
intents.message_content = True  # 視你有沒有需要讀訊息內容

bot = commands.Bot(command_prefix="!", intents=intents)


# ========== 提供給 API 呼叫的發送函式 ==========
async def send_message_to_channel(content: str, channel_id: int | None = None):
    """在指定頻道發送訊息"""
    target_channel_id = channel_id or CHANNEL_ID

    channel = bot.get_channel(target_channel_id)
    if channel is None:
        # 保險一點，抓不到就用 fetch_channel
        channel = await bot.fetch_channel(target_channel_id)

    await channel.send(content)


# ========== HTTP API 部分（aiohttp） ==========
async def handle_send(request: web.Request) -> web.Response:
    """
    POST /send
    JSON body:
    {
        "message": "要發送的內容",
        "channel_id": 123456789012345678   # return web.json_response(
    {"message": "中文測試"},
    dumps=lambda x: json.dumps(x, ensure_ascii=False)
)
可選
    }
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "JSON 格式錯誤"}, status=400)

    message = data.get("message")
    channel_id = data.get("channel_id")

    if not message:
        return web.json_response({"error": "缺少必填欄位 'message'"}, status=400)

    # 如果有傳 channel_id，就轉成 int（可能會是字串）
    if channel_id is not None:
        try:
            channel_id = int(channel_id)
        except ValueError:
            return web.json_response({"error": "channel_id 必須是整數"}, status=400)

    # 把發送任務丟到 bot 的 event loop 去跑
    bot.loop.create_task(send_message_to_channel(message, channel_id))
    print(message)
    return web.json_response(
        {"message": "中文測試"},
        dumps=lambda x: json.dumps(x, ensure_ascii=False)
    )


def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/send", handle_send)
    return app


# ========== Bot 事件 & 啟動程式 ==========
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")


async def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("請先在環境變數中設定 DISCORD_TOKEN")

    # 建立 aiohttp app
    app = create_web_app()
    runner = web.AppRunner(app)
    await runner.setup()

    # 在 localhost:8081 開 HTTP server
    site = web.TCPSite(runner, "0.0.0.0", 8061)
    await site.start()
    print("HTTP API server running on http://localhost:8061")

    # 啟動 Discord Bot
    async with bot:
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
