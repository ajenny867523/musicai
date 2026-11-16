---
title: AI Music Studio
emoji: 🎵
colorFrom: purple
colorTo: indigo
sdk: static
pinned: false
---

# 🎵 AI Music Studio · 前端作業專案

基於 **YuE2-3B** 音樂生成大模型與 **Qwen3** 智能作詞模型打造的獨立 Web 前端介面。
採用純前端架構（HTML5 / CSS3 / ES Module JavaScript），並透過 Hugging Face 官方 `@gradio/client` 連線至後端模型伺服器進行非同步音訊運算。

## 📁 專案前端結構
- **`index.html`**：語意化 HTML5 網頁結構（包含創作面板、動態聲波 Header、即時播放器與樂譜展示）。
- **`style.css`**：現代黑曜石錄音室深色主題（Glassmorphism 毛玻璃、漸層流光、動態聲波跳動特效）。
- **`script.js`**：前端互動邏輯、快速風格按鈕填入、以及透過 `@gradio/client` 調用 YuE2-3B API。
- **`app.py`**（備用）：若老師要求使用 Python 後端環境，此檔案已備妥完整的 Gradio 與 ZeroGPU 程式碼。
