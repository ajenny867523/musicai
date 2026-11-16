# 🎨 網頁外觀客製化指南 (Web UI Customization Guide)

本專案將**邏輯程式碼 (`app.py`)** 與**外觀樣式 (`style.css`)** 完全解耦，讓您可以隨心所欲自由設計與打造屬於自己的介面風格！

---

## 📁 核心檔案結構
- 🎨 **`style.css`**：專門負責所有視覺樣式。包含色彩、字體、背景、卡片陰影、玻璃擬態（Glassmorphism）與按鈕動態效果。
- ⚙️ **`app.py`**：核心後端推論邏輯與 Gradio 佈局，調用與 `mrfakename/yue2-3b` 相同的 **YuE2-3B** 系列模型。
- 📦 **`requirements.txt`** & **`pre-requirements.txt`**：環境依賴配置。
- 📑 **`README.md`**：Hugging Face Space 的配置設定檔（包含 `sdk: gradio`, `suggested_hardware: a10g-large` 與模型關聯標籤）。

---

## 🛠️ 如何自訂外觀？

### 1. 修改主題配色（最簡單快速）
在 `style.css` 的最上方 `:root` 中，您只要調整色碼即可快速切換整體氛圍：

```css
:root {
    /* 🌟 主題配色 - 換成您喜歡的 HEX 色碼 */
    --primary-color: #6366f1;         /* 主色調 (如：科技藍 #2563eb、活力橘 #f97316、賽博朋克粉 #ec4899) */
    --secondary-color: #a855f7;       /* 漸層次色 */
    --accent-color: #06b6d4;          /* 標籤強調色 */
    
    /* 🌌 背景與層次 */
    --bg-main: #0b0f19;               /* 主背景 (深黑、深海藍或純白) */
    --bg-card: rgba(23, 27, 44, 0.85);/* 半透明卡片底色 */
    --bg-border: rgba(255, 255, 255, 0.1); /* 邊框線條 */
}
```

### 2. 幾種推薦配色靈感 (可直接複製貼入 `style.css`)
- **Cyberpunk 霓虹風格**：
  ```css
  --primary-color: #ff007f;
  --secondary-color: #00f0ff;
  --accent-color: #ffe600;
  --bg-main: #05050d;
  ```
- **Studio 專業錄音室暗黑風格**：
  ```css
  --primary-color: #10b981;
  --secondary-color: #059669;
  --accent-color: #34d399;
  --bg-main: #121212;
  --bg-card: #1e1e1e;
  ```
- **Sunset 日落暖調風格**：
  ```css
  --primary-color: #f43f5e;
  --secondary-color: #fb923c;
  --accent-color: #fbbf24;
  --bg-main: #180d19;
  ```

### 3. 修改頂部標題與頁首 (Header)
在 `app.py` 內的 `gr.HTML(...)` 中：
```html
<div class="studio-header">
    <div class="studio-title">🎵 您的專屬音樂工作室名稱</div>
    <div class="studio-subtitle">這裡可以放您自己設計的歡迎詞與介紹！</div>
</div>
```

---

## 🚀 如何同步更新至 Hugging Face Space？
當您修改完成後，在終端機中執行：
```bash
git add .
git commit -m "Update custom UI design"
git push origin main
```
Hugging Face 便會自動偵測變更並以 ZeroGPU 重新建置您的自訂音樂生成空間！
