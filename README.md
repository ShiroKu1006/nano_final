# Intruder Detection & Discord Alert System

本專案包含兩個主要模組：

1. Discord Bot + HTTP API 服務
   - 提供 REST API，可由外部程式觸發 Discord 訊息與檔案傳送
2. 即時入侵者偵測系統
   - 使用 YOLO + TensorRT 進行即時人物偵測
   - 偵測到入侵者時，透過 API 呼叫 Discord Bot 發送警報
  
專案架構
```
.
├── discord_main.py          # Discord Bot + HTTP API
├── intruder_detection.py    # 入侵者即時偵測主程式
├── requirements.txt
└── .env
```

## Discord Bot 與 HTTP API（discord_main.py）
### 📌 功能說明
- 使用 discord.py 建立 Discord Bot
- 使用 aiohttp 提供 HTTP POST API
- 支援：
  - 發送純文字訊息
  - 發送文字 + 檔案
  - 指定 Discord 頻道 ID
### 🔧 環境變數設定（.env）
```
DISCORD_TOKEN=你的 Discord Bot Token
CHANNEL_ID=預設頻道 ID
```
### 📡 API 介面說明
```
{
  URL：POST /send
  Port：8061
  Content-Type：application/json
  Request Body 範例
}
```
## 主流程概述

1. 攝影機擷取畫面
2. YOLO TensorRT 推論
3. 後處理（NMS、confidence 過濾）
4. 偵測到人員：
   - 每 10 秒發送一次 Discord 警告
   - 擷取畫面
   - 開始或延續錄影
5. 人員消失後：
   - 停止錄影
   - 重置警告計時器
