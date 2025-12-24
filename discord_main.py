import os
import asyncio
import json
import discord
from discord.ext import commands
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 支援發送文字與檔案
async def send_message_to_channel(content: str, file_path: str = None, channel_id: int = None):
    target_channel_id = channel_id or CHANNEL_ID
    channel = bot.get_channel(target_channel_id)
    if channel is None:
        channel = await bot.fetch_channel(target_channel_id)

    if file_path and os.path.exists(file_path):
        # 建立 discord.File 物件
        d_file = discord.File(file_path)
        await channel.send(content, file=d_file)
    else:
        await channel.send(content)

async def handle_send(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "JSON 格式錯誤"}, status=400)

    message = data.get("message")
    file_path = data.get("file_path")  # 新增：接收檔案路徑
    channel_id = data.get("channel_id")

    if not message and not file_path:
        return web.json_response({"error": "缺少 message 或 file_path"}, status=400)

    if channel_id is not None:
        try:
            channel_id = int(channel_id)
        except ValueError:
            return web.json_response({"error": "channel_id 必須是整數"}, status=400)

    # 執行任務
    asyncio.create_task(send_message_to_channel(message, file_path, channel_id))
    
    return web.json_response(
        {"status": "success", "file": file_path},
        dumps=lambda x: json.dumps(x, ensure_ascii=False)
    )

def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/send", handle_send)
    return app

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

async def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN 未設定")

    app = create_web_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8061)
    await site.start()

    async with bot:
        await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())