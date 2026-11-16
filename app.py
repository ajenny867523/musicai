import os
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import gradio as gr
import spaces
import torch
from qwen_asr import Qwen3ASRModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from yue2 import YuE2Pipeline

# ==============================================================================
# 🤖 模型識別碼 (Models Configuration) - 與 mrfakename/yue2-3b 完全一致
# ==============================================================================
MODEL_ID = "m-a-p/YuE2-3B"
TRANSCRIBER_ID = "m-a-p/SheetSage2"
LYRIC_MODEL_ID = "Qwen/Qwen3-4B"
ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# ------------------------------------------------------------------------------
# 🚀 載入模型 (Model Initialization)
# ZeroGPU 在啟動時會模擬 CUDA，透過 spaces.GPU 在呼叫時動態分配實際 GPU 運算資源。
# ------------------------------------------------------------------------------
set_memory_fraction = torch.cuda.set_per_process_memory_fraction
try:
    # 略過單進程 CUDA 記憶體限制以適應 ZeroGPU 模擬環境
    torch.cuda.set_per_process_memory_fraction = lambda *args, **kwargs: None
    pipe = YuE2Pipeline.from_pretrained(
        MODEL_ID,
        device="cuda",
        backend="torch",
        progress=False,
    )
finally:
    torch.cuda.set_per_process_memory_fraction = set_memory_fraction

transcriber = AutoModel.from_pretrained(
    TRANSCRIBER_ID,
    trust_remote_code=True,
).eval().to("cuda")

lyric_tokenizer = AutoTokenizer.from_pretrained(LYRIC_MODEL_ID)
lyric_model = AutoModelForCausalLM.from_pretrained(
    LYRIC_MODEL_ID,
    torch_dtype=torch.bfloat16,
).eval().to("cuda")

asr_model = Qwen3ASRModel.from_pretrained(
    ASR_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=1,
    max_new_tokens=2048,
)

# ==============================================================================
# ⚙️ 輔助函式 (Helper Functions)
# ==============================================================================
def estimate_duration(style, lyrics, planning_mode, render_quality, seed):
    del style, planning_mode, render_quality, seed
    # 根據歌詞長度動態調整預估排隊時間
    return min(300, max(120, 120 + len(lyrics or "") // 12))


def validate_inputs(style, lyrics):
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    if not style:
        raise gr.Error("請填寫音樂風格描述 (Describe the musical style)。")
    if not lyrics:
        raise gr.Error("請填寫歌詞並附帶段落標籤，如 [Verse] 與 [Chorus]。")
    if len(style) > 1_000:
        raise gr.Error("風格描述長度請控制在 1,000 字以內。")
    if len(lyrics) > 12_000:
        raise gr.Error("歌詞長度請控制在 12,000 字以內。")
    return style, lyrics


def save_result(song, prefix):
    output_dir = Path(tempfile.mkdtemp(prefix=prefix))
    flac_path = output_dir / "song.flac"
    mp3_path = output_dir / "song.mp3"
    song.save(flac_path)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(flac_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(mp3_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise gr.Error(f"歌曲已生成，但 MP3 轉碼失敗: {detail}") from exc
    score = song.abc or "直接生成模式未產生符號樂譜。"
    return str(mp3_path), str(flac_path), score


def took_since(t0):
    return f"⚡ 生成耗時: {time.time() - t0:.1f} 秒"


# ==============================================================================
# 🎯 核心運算任務 (Inference Tasks with ZeroGPU)
# ==============================================================================
@spaces.GPU(duration=240)
def analyze_cover_source(audio_path, language):
    if not audio_path:
        raise gr.Error("請先上傳原始歌曲音訊。")
    try:
        result = transcriber.transcribe(audio_path, melody_only=True)
        asr_result = asr_model.transcribe(
            audio=audio_path,
            language=None if language == "Auto-detect" else language,
        )
        transcript = asr_result[0].text.strip()
    except Exception as exc:
        raise gr.Error(f"來源歌曲分析失敗: {exc}") from exc
    abc_score = result.get("abc")
    if not abc_score:
        raise gr.Error("SheetSage2 未能解析出此音訊的 ABC 旋律譜。")
    if not transcript:
        raise gr.Error("Qwen3-ASR 未在此音訊中辨識出歌詞。")
    return (
        abc_score,
        transcript,
        "✅ 旋律與歌詞分析完成！請於下方檢視、標記段落後進行生成。",
    )


@spaces.GPU(duration=90)
def write_lyrics(concept, style, language, structure, seed):
    concept = (concept or "").strip()
    style = (style or "").strip()
    if not concept:
        raise gr.Error("請描述歌曲概念或主題故事。")
    prompt = f"""Write original song lyrics for a music-generation model.

Concept or story: {concept}
Musical style: {style or 'unspecified'}
Language: {language}
Requested structure: {structure}

Return only the finished lyrics. Use explicit section labels such as [Verse], [Pre-Chorus],
[Chorus], [Bridge], [Instrumental], and [Outro]. Keep lines singable, use a memorable chorus,
and avoid commentary, markdown fences, titles, or production instructions."""

    messages = [{"role": "user", "content": prompt}]
    inputs = lyric_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(lyric_model.device)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        output_ids = lyric_model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    lyrics = lyric_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    if not lyrics:
        raise gr.Error("歌詞生成結果為空，請嘗試提供更具體的概念描述。")
    return lyrics


@spaces.GPU(duration=estimate_duration)
def generate_song(style, lyrics, planning_mode, render_quality, seed):
    t0 = time.time()
    style, lyrics = validate_inputs(style, lyrics)

    original_config = pipe.generation_config
    pipe.generation_config = replace(original_config, ode_steps=int(render_quality))
    try:
        song = pipe(
            style=style,
            lyrics=lyrics,
            cot=planning_mode,
            seed=int(seed),
        )
    except Exception as exc:
        raise gr.Error(f"歌曲生成失敗: {exc}") from exc
    finally:
        pipe.generation_config = original_config

    return (*save_result(song, "yue2-generation-"), took_since(t0))


@spaces.GPU(duration=300)
def generate_cover(style, lyrics, abc_score, render_quality, seed):
    t0 = time.time()
    style, lyrics = validate_inputs(style, lyrics)
    abc_score = (abc_score or "").strip()
    if not abc_score:
        raise gr.Error("請提供來源歌曲的旋律 ABC 譜。")
    if len(abc_score) > 100_000:
        raise gr.Error("ABC 樂譜長度請控制在 100,000 字以內。")

    original_config = pipe.generation_config
    pipe.generation_config = replace(original_config, ode_steps=int(render_quality))
    try:
        song = pipe(
            style=style,
            lyrics=lyrics,
            abc=abc_score,
            cot="melody",
            seed=int(seed),
        )
    except Exception as exc:
        raise gr.Error(f"翻唱歌曲生成失敗: {exc}") from exc
    finally:
        pipe.generation_config = original_config

    return (*save_result(song, "yue2-cover-"), took_since(t0))


# ==============================================================================
# 💡 範例資料 (Examples)
# ==============================================================================
EXAMPLES = [
    [
        "Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths, uplifting",
        """[Verse]\nCity windows turn to gold\nEvery streetlight has a story\nWe are brave and we are bold\nRunning toward the morning glory\n\n[Chorus]\nStay awake, the night is ours\nWe can dance beneath the stars\nHold this moment, hold it tight\nWe are sparks inside the night\n\n[Outro]\nInside the night""",
        "full",
        16,
        42,
    ],
    [
        "Acoustic indie folk, intimate male vocal, fingerpicked guitar, gentle strings",
        """[Verse]\nDust is dancing in the doorway\nSummer settles on the road\nI can hear the old trees whisper\nAll the secrets that they know\n\n[Chorus]\nTake me home across the river\nWhere the evening moves so slow\nIf the wind can find its way there\nThen I know that I can go""",
        "melody",
        16,
        831001,
    ],
]


# ==============================================================================
# 🎨 網頁介面佈局 (Web UI Layout)
# 支援 style.css 自由客製化樣式！
# ==============================================================================
css_file = Path(__file__).parent / "style.css"
custom_css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""

with gr.Blocks(title="AI Music Studio · YuE2-3B", css=custom_css) as demo:
    # 🌟 自訂頁首區塊 (Header)
    gr.HTML("""
    <div class="studio-header">
        <div class="studio-title">🎵 AI Music Studio</div>
        <div class="studio-subtitle">
            基於 <b>YuE2-3B</b> 全歌曲開源音樂大模型，輸入風格描述與歌詞即刻生成高品質完整歌曲或全新翻唱！
        </div>
        <div class="model-badges">
            <span class="badge">🔥 YuE2-3B Foundation</span>
            <span class="badge">🎼 SheetSage2 Melody</span>
            <span class="badge">✍️ Qwen3-4B Lyricist</span>
            <span class="badge">🎙️ Qwen3-ASR Transcriber</span>
            <span class="badge">⚡ ZeroGPU Powered</span>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):
        # ----------------------------------------------------------------------
        # Tab 1: 創作歌曲 (Create Song)
        # ----------------------------------------------------------------------
        with gr.Tab("✨ 創作歌曲 (Create Song)"):
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["studio-card"]):
                    style = gr.Textbox(
                        label="音樂風格與氛圍 (Style Prompt)",
                        placeholder="例如：Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths...",
                        value="Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths",
                        lines=2,
                    )
                    
                    with gr.Accordion("🪄 使用 Qwen3-4B 智能寫詞 (AI Lyric Assistant)", open=False):
                        lyric_concept = gr.Textbox(
                            label="歌曲靈感與主題故事 (Song Concept)",
                            placeholder="例如：夏日深夜在海邊奔馳的公路旅行，關於青春與告別的故事...",
                        )
                        with gr.Row():
                            lyric_language = gr.Dropdown(
                                choices=["English", "Chinese", "Cantonese", "Japanese", "Korean", "Spanish"],
                                value="Chinese",
                                label="歌詞語言 (Language)",
                            )
                            lyric_structure = gr.Dropdown(
                                choices=[
                                    "Verse – Chorus – Verse – Chorus – Bridge – Chorus – Outro",
                                    "Verse – Pre-Chorus – Chorus – Verse – Pre-Chorus – Chorus – Bridge – Chorus",
                                    "Intro – Verse – Chorus – Instrumental – Verse – Chorus – Outro",
                                ],
                                value="Verse – Chorus – Verse – Chorus – Bridge – Chorus – Outro",
                                label="段落架構 (Structure)",
                            )
                        lyric_seed = gr.Number(value=42, precision=0, label="詞曲隨機種子 (Seed)")
                        write_lyrics_button = gr.Button("📝 一鍵生成歌詞", elem_classes=["secondary-action-btn"])
                    
                    lyrics = gr.Textbox(
                        label="段落歌詞 (Lyrics - 請務必使用 [Verse], [Chorus] 等標籤)",
                        lines=14,
                        placeholder="[Verse]\n在微涼的晚風中...\n\n[Chorus]\n奔向未知的彼方...",
                    )
                    
                    with gr.Row():
                        planning_mode = gr.Radio(
                            choices=[("旋律 + 和弦 (Melody + chords)", "full"), ("純旋律 (Melody only)", "melody"), ("不生成樂譜 (No score)", "off")],
                            value="full",
                            label="符號規劃模式 (Symbolic Planning)",
                        )
                        render_quality = gr.Radio(
                            choices=[("快速 · 16 步 (Fast)", 16), ("最佳品質 · 32 步 (Best)", 32)],
                            value=16,
                            label="彩現品質 (Render Quality)",
                        )
                        seed = gr.Number(value=42, precision=0, label="音樂隨機種子 (Seed)")

                    generate_btn = gr.Button("🚀 開始生成音樂 (Generate Song)", elem_classes=["primary-action-btn"])

                with gr.Column(scale=2, elem_classes=["studio-card"]):
                    audio_output = gr.Audio(label="生成音訊 (MP3 預覽)", type="filepath", format="mp3")
                    duration_badge = gr.Markdown("", elem_classes=["time-badge"])
                    flac_download = gr.File(label="下載無損音質 (Lossless FLAC)")
                    with gr.Accordion("📜 符號樂譜 (ABC Score)", open=False):
                        score_output = gr.Code(language=None, label="可編輯 ABC 樂譜")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[style, lyrics, planning_mode, render_quality, seed],
                cache_examples=False,
            )

        # ----------------------------------------------------------------------
        # Tab 2: 歌曲翻唱與改編 (AI Cover)
        # ----------------------------------------------------------------------
        with gr.Tab("🎧 歌曲翻唱與改編 (AI Cover)"):
            gr.Markdown(
                "上傳歌曲音訊，透過 **SheetSage2** 辨識旋律並以 **Qwen3-ASR** 轉錄歌詞，"
                "接著您可以調整樂譜或歌詞，並指定全新曲風進行 AI 翻唱！"
            )
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["studio-card"]):
                    source_audio = gr.Audio(
                        label="來源歌曲音訊 (Source Song)",
                        type="filepath",
                        sources=["upload"],
                    )
                    asr_language = gr.Dropdown(
                        choices=["Auto-detect", "English", "Chinese", "Cantonese", "Japanese", "Korean", "Spanish"],
                        value="Auto-detect",
                        label="歌詞辨識語言 (Lyrics Language)",
                    )
                    transcribe_button = gr.Button("🔍 1. 分析並轉錄旋律與歌詞", elem_classes=["secondary-action-btn"])
                    transcription_status = gr.Markdown()
                    
                    cover_style = gr.Textbox(
                        label="目標改編風格 (New Style)",
                        placeholder="例如：Jazz-funk, warm lead vocal, Rhodes piano, tight drums...",
                    )
                    cover_lyrics = gr.Textbox(
                        label="對齊歌詞 (Matched Lyrics - 請包含段落標籤)",
                        lines=10,
                        placeholder="[Verse]\n依據原曲段落排版的歌詞...",
                    )
                    cover_abc = gr.Textbox(
                        label="旋律 ABC 樂譜 (可在生成前微調音符)",
                        lines=10,
                        placeholder="X:1\nT:Source melody\nM:4/4\nL:1/8\nK:C\n...",
                    )
                    with gr.Row():
                        cover_render_quality = gr.Radio(
                            choices=[("快速 · 16 步 (Fast)", 16), ("最佳品質 · 32 步 (Best)", 32)],
                            value=16,
                            label="彩現品質 (Render Quality)",
                        )
                        cover_seed = gr.Number(value=831001, precision=0, label="隨機種子 (Seed)")
                    
                    cover_button = gr.Button("🎉 2. 生成全新風格翻唱 (Generate Cover)", elem_classes=["primary-action-btn"])

                with gr.Column(scale=2, elem_classes=["studio-card"]):
                    cover_audio = gr.Audio(label="翻唱成果 (MP3 預覽)", type="filepath", format="mp3")
                    cover_duration_badge = gr.Markdown("", elem_classes=["time-badge"])
                    cover_flac_download = gr.File(label="下載無損音質 (Lossless FLAC)")
                    with gr.Accordion("📜 翻唱 ABC 樂譜", open=False):
                        cover_score = gr.Code(language=None, label="使用樂譜")

    # 🌟 自訂頁尾區塊 (Footer)
    gr.HTML("""
    <div class="studio-footer">
        YuE2-3B 採用 CC BY-NC 4.0 授權條款 · 完整支援 ZeroGPU 加速 · 介面支援自訂 style.css 擴充
    </div>
    """)

    # --------------------------------------------------------------------------
    # 事件綁定 (Event Listeners)
    # --------------------------------------------------------------------------
    generate_btn.click(
        fn=generate_song,
        inputs=[style, lyrics, planning_mode, render_quality, seed],
        outputs=[audio_output, flac_download, score_output, duration_badge],
        api_name="generate_song",
    )
    write_lyrics_button.click(
        fn=write_lyrics,
        inputs=[lyric_concept, style, lyric_language, lyric_structure, lyric_seed],
        outputs=[lyrics],
    )
    cover_button.click(
        fn=generate_cover,
        inputs=[cover_style, cover_lyrics, cover_abc, cover_render_quality, cover_seed],
        outputs=[cover_audio, cover_flac_download, cover_score, cover_duration_badge],
    )
    transcribe_button.click(
        fn=analyze_cover_source,
        inputs=[source_audio, asr_language],
        outputs=[cover_abc, cover_lyrics, transcription_status],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None, max_size=50).launch()
