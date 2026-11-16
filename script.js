/**
 * AI Music Studio - 前端主程式邏輯
 * 採用標準 ES Module 格式，透過 Hugging Face 官方 @gradio/client 直接與後端 AI 模型串接
 */

import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

// 🎯 目標後端 Space：直接連線至公開的 YuE2-3B ZeroGPU 伺服器
const TARGET_SPACE = "mrfakename/yue2-3b";

// 全域客戶端實例快取
let clientInstance = null;

async function getClient() {
    if (!clientInstance) {
        updateStatus("正在建立與 YuE2-3B 伺服器的安全 WebSocket 連線...");
        clientInstance = await Client.connect(TARGET_SPACE);
    }
    return clientInstance;
}

// =============================================================================
// DOM 元素選取
// =============================================================================
// 導覽頁籤
const navTabs = document.querySelectorAll(".nav-tab");
const tabPanes = document.querySelectorAll(".tab-pane");

// 創作頁面元素
const styleInput = document.getElementById("styleInput");
const lyricsInput = document.getElementById("lyricsInput");
const seedInput = document.getElementById("seedInput");
const presetChips = document.querySelectorAll(".preset-chip");
const btnGenerateSong = document.getElementById("btnGenerateSong");

// 歌詞助理手風琴
const lyricAccordionBtn = document.getElementById("lyricAccordionBtn");
const lyricAccordionBody = document.getElementById("lyricAccordionBody");
const lyricConcept = document.getElementById("lyricConcept");
const lyricLanguage = document.getElementById("lyricLanguage");
const lyricStructure = document.getElementById("lyricStructure");
const btnWriteLyrics = document.getElementById("btnWriteLyrics");

// 結果展示元素
const emptyAudioState = document.getElementById("emptyAudioState");
const activePlayerBox = document.getElementById("activePlayerBox");
const audioPlayer = document.getElementById("audioPlayer");
const tookBadge = document.getElementById("tookBadge");
const btnDownloadMp3 = document.getElementById("btnDownloadMp3");
const btnDownloadFlac = document.getElementById("btnDownloadFlac");
const abcScoreBox = document.getElementById("abcScoreBox");
const btnCopyScore = document.getElementById("btnCopyScore");
const statusBox = document.getElementById("statusBox");
const statusText = document.getElementById("statusText");

// =============================================================================
// UI 互動事件綁定
// =============================================================================

// 1. 分頁切換 (Tab Navigation)
navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
        navTabs.forEach(t => t.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));

        tab.classList.add("active");
        const targetTabId = tab.getAttribute("data-tab");
        const targetPane = document.getElementById(targetTabId);
        if (targetPane) targetPane.classList.add("active");
    });
});

// 2. 快速風格標籤 (Preset Chips) 點擊自動填入
presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
        const styleText = chip.getAttribute("data-style");
        if (styleText) {
            styleInput.value = styleText;
            styleInput.focus();
            // 給予按鈕短暫的高亮反饋
            chip.style.transform = "scale(0.96)";
            setTimeout(() => { chip.style.transform = ""; }, 150);
        }
    });
});

// 3. 智能作詞手風琴收合/展開
if (lyricAccordionBtn && lyricAccordionBody) {
    lyricAccordionBtn.addEventListener("click", () => {
        const isOpen = lyricAccordionBody.classList.contains("open");
        if (isOpen) {
            lyricAccordionBody.classList.remove("open");
            lyricAccordionBtn.querySelector(".chevron").textContent = "▼";
        } else {
            lyricAccordionBody.classList.add("open");
            lyricAccordionBtn.querySelector(".chevron").textContent = "▲";
        }
    });
}

// 4. 複製 ABC 樂譜
if (btnCopyScore) {
    btnCopyScore.addEventListener("click", () => {
        const score = abcScoreBox.textContent;
        if (!score || score.includes("等待歌曲生成中")) {
            alert("目前尚無可複製的樂譜！");
            return;
        }
        navigator.clipboard.writeText(score).then(() => {
            const originalText = btnCopyScore.textContent;
            btnCopyScore.textContent = "✅ 已複製！";
            setTimeout(() => { btnCopyScore.textContent = originalText; }, 2000);
        });
    });
}

// 狀態更新輔助函式
function updateStatus(text, show = true) {
    if (!statusBox) return;
    if (show) {
        statusBox.classList.remove("hidden");
        statusText.textContent = text;
    } else {
        statusBox.classList.add("hidden");
    }
}

// =============================================================================
// AI 模型推論呼叫 (Backend API Calls)
// =============================================================================

// 🚀 1. 開始生成歌曲 (Generate Song)
if (btnGenerateSong) {
    btnGenerateSong.addEventListener("click", async () => {
        const style = styleInput.value.trim();
        const lyrics = lyricsInput.value.trim();
        const seed = parseInt(seedInput.value) || 42;

        const planningRadio = document.querySelector('input[name="planningMode"]:checked');
        const planningMode = planningRadio ? planningRadio.value : "full";

        const qualityRadio = document.querySelector('input[name="renderQuality"]:checked');
        const renderQuality = qualityRadio ? parseInt(qualityRadio.value) : 16;

        if (!style) {
            alert("請填寫音樂風格描述！");
            styleInput.focus();
            return;
        }
        if (!lyrics) {
            alert("請填寫歌詞並附帶段落標籤（如 [Verse], [Chorus]）！");
            lyricsInput.focus();
            return;
        }

        // 切換按鈕與狀態至 Loading
        btnGenerateSong.disabled = true;
        const btnText = btnGenerateSong.querySelector(".btn-text");
        const spinner = btnGenerateSong.querySelector(".spinner");
        if (btnText) btnText.textContent = "⏳ 正在生成音樂 (約需 1~2 分鐘)...";
        if (spinner) spinner.classList.remove("hidden");

        updateStatus("正在連接 YuE2-3B 模型，準備進行音訊渲染...");

        try {
            const client = await getClient();
            updateStatus("✨ 已進入 ZeroGPU 動態排程，正在渲染人聲與配器伴奏...");

            // 調用 Gradio 後端的 /generate_song API
            const result = await client.predict("/generate_song", [
                style,
                lyrics,
                planningMode,
                renderQuality,
                seed
            ]);

            console.log("生成結果：", result);

            // result.data 格式：[mp3_obj_or_url, flac_obj_or_url, abc_score, took_text]
            const [mp3Data, flacData, scoreData, tookText] = result.data;

            // 取得音訊 URL
            const mp3Url = typeof mp3Data === "string" ? mp3Data : mp3Data?.url;
            const flacUrl = typeof flacData === "string" ? flacData : flacData?.url;

            if (mp3Url) {
                audioPlayer.src = mp3Url;
                audioPlayer.load();
                btnDownloadMp3.href = mp3Url;
                if (flacUrl) btnDownloadFlac.href = flacUrl;

                tookBadge.textContent = tookText || "⚡ 生成完成";
                abcScoreBox.textContent = scoreData || "直接生成模式未產生符號樂譜。";

                // 切換展示面板
                emptyAudioState.classList.add("hidden");
                activePlayerBox.classList.remove("hidden");
                audioPlayer.play().catch(() => {}); // 嘗試自動播放
            } else {
                alert("音訊生成完成，但未取得音訊連結。");
            }
        } catch (error) {
            console.error("生成失敗：", error);
            alert("生成過程中發生錯誤：" + (error.message || error));
        } finally {
            btnGenerateSong.disabled = false;
            if (btnText) btnText.textContent = "🚀 開始生成歌曲 (Generate Song)";
            if (spinner) spinner.classList.add("hidden");
            updateStatus("", false);
        }
    });
}

// 🪄 2. 使用 Qwen3-4B 生成結構化歌詞
if (btnWriteLyrics) {
    btnWriteLyrics.addEventListener("click", async () => {
        const concept = lyricConcept.value.trim();
        const style = styleInput.value.trim();
        const language = lyricLanguage.value;
        const structure = lyricStructure.value;
        const seed = 42;

        if (!concept) {
            alert("請先輸入歌曲靈感或故事主題！");
            lyricConcept.focus();
            return;
        }

        btnWriteLyrics.disabled = true;
        btnWriteLyrics.textContent = "⏳ 正在構思歌詞中...";

        try {
            const client = await getClient();
            // 調用寫詞函式 (fn_index 1)
            const result = await client.predict(1, [
                concept,
                style,
                language,
                structure,
                seed
            ]);

            if (result.data && result.data[0]) {
                lyricsInput.value = result.data[0];
                alert("✅ 歌詞已成功生成並填入下方歌詞框！");
            }
        } catch (error) {
            console.error("作詞失敗：", error);
            alert("智能作詞失敗：" + (error.message || error));
        } finally {
            btnWriteLyrics.disabled = false;
            btnWriteLyrics.textContent = "📝 生成結構化歌詞";
        }
    });
}
