# 🎨 AI Music Studio 網頁外觀客製化手冊

本指南說明整個專案的前端設計架構，以及為何此專案使用 `app.py + style.css` 而非傳統的 `index.html`。

---

## ❓ 常見疑問：為何專案中沒有 `index.html`？

### 1. 靜態網頁（Static） vs. 機器學習應用（Gradio / Python）
- **過去的靜態網頁（Static Space）**：只包含 HTML/CSS/JS，瀏覽器直接開啟 `index.html` 呈現畫面。但**純 HTML 網頁無法在伺服器端載入 30 億參數的 AI 模型（如 YuE2-3B、SheetSage2、Qwen3 等），也無法使用 Hugging Face 的 ZeroGPU 顯卡**。
- **現在的 Gradio 應用（Gradio Space）**：
  - `app.py` 是整個網站的伺服器與後端入口。
  - Gradio 會在後端自動為您編譯並生成現代化的前端網頁 DOM 結構，並建立與 GPU 伺服器的高速 WebSocket 串流通訊。
  - **在 Gradio 架構下，不需要、也不能放置靜態的 `index.html` 作為主入口**（否則 Hugging Face 會找不到 Python 應用的進入點）。

### 2. 那我想自己寫 HTML 該怎麼辦？
您完全可以在 `app.py` 中使用 `gr.HTML(""" 您的 HTML 代碼 """)` 嵌入任何標準 HTML 標籤！例如：
- 頂部的標題與視覺橫幅
- 音訊動態跳動的聲波特效（目前已實作於頁首）
- 任何自訂的按鈕、提示卡片、圖片或文字說明
所有嵌入的 HTML 元素都可以直接在 `style.css` 透過 class 或 id 進行百分之百的視覺設計！

---

## 🛠️ 如何調整與客製化現有設計？

### 1. 調整主題配色與視覺光暈（開啟 `style.css`）
在 `style.css` 最上方的 `:root` 變數區，您可以直接更換色碼：

```css
:root {
    /* 漸層品牌主色 (支援 CSS 漸層) */
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    --primary-color: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-pink: #ec4899;

    /* 深邃背景色 */
    --bg-page: #0a0d14;                  /* 網頁底色 */
    --bg-card: rgba(18, 24, 38, 0.78);   /* 半透明卡片底色 */
    --border-subtle: rgba(255, 255, 255, 0.08); /* 細緻邊框 */
}
```

### 2. 調整按鈕或卡片圓角
- 想要更圓潤柔和的風格？修改 `--radius-md: 20px;` 與 `--radius-lg: 28px;`。
- 想要更俐落科技的直角風格？修改 `--radius-md: 4px;`。

### 3. 新增或修改快速風格標籤 (Quick Presets)
在 `app.py` 中的 `PRESET_STYLES` 陣列，您可以自由增加喜歡的音樂曲風：
```python
PRESET_STYLES = [
    ("🌆 80s 復古合成波", "...提示詞..."),
    ("☕ 慵懶 Lo-Fi", "...提示詞..."),
    ("🎸 獨立民謠", "...提示詞..."),
    # 可以新增您自己的專屬曲風！
]
```

---

## 🚀 同步至 Hugging Face Space
當您完成調整後，只需在終端機中執行：
```bash
git add .
git commit -m "Update studio design"
git push origin main
```
Hugging Face 便會自動重新打包並部署您設計的全新外觀！
