# 🤖 AI Investment Assistant
### Trading 212 × Telegram — UK小額投資自動助理

> **預算 £300 | 每次交易前通知確認 | 免佣金交易**

---

## ⚠️ 重要免責聲明

> 本工具僅供學習用途。股票投資有風險，可能損失全部本金。  
> **請先在 Demo（模擬）帳戶測試至少2週，確認系統正常後再切換到真實帳戶。**  
> 本工具不提供財務建議，所有投資決策由你自行負責。

---

## 📋 系統概覽

```
每4小時自動分析市場
       ↓
發現買入機會（RSI + 移動平均線 + 動能）
       ↓
Telegram 推送通知到你的手機
       ↓
你回覆 YES 或 NO
       ↓
YES → 在 Trading 212 下單
NO  → 跳過，等待下一次機會
```

---

## 🛠️ 第一步：開立 Trading 212 帳戶

1. 下載 **Trading 212** App（iOS / Android）
2. 點選 **"Open Account"** → 選擇 **Invest**（股票帳戶）
3. 完成 KYC 驗證（需要護照/ID + 地址証明）
4. 建議先開啟 **Practice Account（模擬帳戶）** 測試

> 🇬🇧 Trading 212 是 FCA 監管的合法英國券商，免佣金

---

## 🔑 第二步：取得 Trading 212 API Key

1. 打開 Trading 212 App
2. 點選右上角頭像 → **Settings**
3. 找到 **API (beta)** 選項
4. 點選 **"Generate API Key"**
5. 記下 **API Key** 和 **API Secret**（只顯示一次！）

> 💡 如果要測試，先在 **Practice Account** 生成 API Key

---

## 🤖 第三步：建立 Telegram Bot

### 3.1 建立 Bot
1. 在 Telegram 搜尋 **@BotFather**
2. 發送 `/newbot`
3. 輸入 Bot 名稱（例如：`My Investment Assistant`）
4. 輸入 Bot 用戶名（例如：`my_invest_bot`，必須以 `_bot` 結尾）
5. 複製得到的 **Token**（格式：`123456789:ABCdef...`）

### 3.2 取得你的 Chat ID
1. 在 Telegram 搜尋你剛建立的 Bot，點 **Start**
2. 發送任意一條訊息（如 `hello`）
3. 在瀏覽器打開：
   ```
   https://api.telegram.org/bot<你的TOKEN>/getUpdates
   ```
4. 找到 `"chat":{"id":XXXXXXXX}` 裡的數字，那就是你的 Chat ID

---

## 💻 第四步：在 Mac 上安裝

### 4.1 安裝 Python（如果沒有）
```bash
# 在 Terminal 輸入：
python3 --version

# 如果沒有，從這裡下載：
# https://www.python.org/downloads/
```

### 4.2 下載並設定助理
```bash
# 把 investment_assistant 資料夾放到桌面
cd ~/Desktop/investment_assistant

# 複製設定檔
cp .env.example .env

# 用文字編輯器開啟設定
open -e .env
```

### 4.3 填寫 .env 設定檔
```env
T212_API_KEY=你的Trading212 API Key
T212_API_SECRET=你的Trading212 API Secret
T212_MODE=demo                    ← 先用 demo！測試OK再改成 live

TELEGRAM_TOKEN=你的Bot Token
TELEGRAM_CHAT_ID=你的Chat ID

TOTAL_BUDGET=300                  ← 你的總預算（英鎊）
MAX_PER_TRADE=30                  ← 每次最多投資金額
CONFIRM_TIMEOUT=300               ← 等待確認秒數（5分鐘）
```

### 4.4 啟動
```bash
chmod +x start.sh
./start.sh
```

---

## 📱 使用方式

### 你會收到這樣的 Telegram 通知：

```
🤖 Investment Assistant – Trade Proposal
🆔 Trade ID: T0001
⏰ 13/05/2025 14:30

📦 Vanguard S&P 500 ETF
   Ticker: VUSA
   Signal: 🟢 BUY
   Confidence: 72%
   Current price: £84.50
   RSI: 38.2
   5-day momentum: +2.1%

📋 Signals:
  • 📈 Price above both MAs (bullish)
  • 🔥 RSI oversold (38.2) - potential buy
  • 🚀 Strong 5-day momentum (+2.1%)

💷 Proposed investment: £25.00

⚡ Reply YES to confirm or NO to skip
⏳ Auto-cancel in 5 minutes
```

### 你的回覆：
| 回覆 | 動作 |
|------|------|
| `YES` | 確認下單 |
| `NO` | 跳過此交易 |
| `Y` / `確認` / `OK` | 確認下單 |
| `N` / `取消` / `CANCEL` | 跳過 |

---

## 📊 投資策略說明

系統使用**技術分析**決定買入時機：

| 指標 | 說明 |
|------|------|
| **移動平均線 (MA20/MA50)** | 判斷趨勢方向 |
| **RSI（相對強弱指數）** | RSI < 30 = 超賣 = 潛在買入機會 |
| **5日動能** | 近期價格走勢 |

### 預設觀察清單：
| 資產 | 類型 | 分配比例 |
|------|------|---------|
| Vanguard S&P 500 (VUSA) | ETF | 30% |
| Vanguard All-World (VWRL) | ETF | 20% |
| iShares FTSE 100 (ISF) | ETF | 10% |
| Apple (AAPL) | 美股 | 15% |
| Microsoft (MSFT) | 美股 | 15% |
| Shell PLC (SHEL) | 英股 | 10% |

---

## 🔄 讓程式在背景持續運行（Mac）

### 方法一：使用 Screen（推薦）
```bash
# 安裝 screen（如果沒有）
brew install screen

# 在背景啟動
screen -S invest
./start.sh

# 離開 screen（程式繼續運行）
Ctrl+A 然後按 D

# 重新連回
screen -r invest
```

### 方法二：設定為開機自動啟動
```bash
# 建立 LaunchAgent
cat > ~/Library/LaunchAgents/com.investment.assistant.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.investment.assistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$HOME/Desktop/investment_assistant/investment_assistant.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$HOME/Desktop/investment_assistant</string>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/Desktop/investment_assistant/assistant.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Desktop/investment_assistant/error.log</string>
</dict>
</plist>
EOF

# 啟動服務
launchctl load ~/Library/LaunchAgents/com.investment.assistant.plist
```

---

## ❓ 常見問題

**Q: 程式安全嗎？**  
A: API Key 只存在你的本地 .env 文件，不會上傳到任何地方。

**Q: Demo 和 Live 有什麼差別？**  
A: Demo = 模擬交易，不用真錢。Live = 真實資金。**強烈建議先用 Demo 跑2週**。

**Q: 可以修改觀察清單嗎？**  
A: 可以，編輯 `investment_assistant.py` 裡的 `WATCHLIST` 清單。

**Q: 如何暫停系統？**  
A: 在 Terminal 按 `Ctrl+C`，或在 screen 裡輸入 `Ctrl+A` 再 `K`。

---

## 📝 日誌查看

```bash
# 查看最新日誌
tail -f assistant.log

# 查看錯誤
cat error.log
```

---

*最後更新：2026年5月 | 適用於 Trading 212 API v0 (beta)*
