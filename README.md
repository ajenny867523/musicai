---
title: YuE2-3B Music Generator
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
python_version: "3.10.13"
suggested_hardware: a10g-large
models:
- m-a-p/YuE2-3B
- m-a-p/YuE2-Vae
- m-a-p/SheetSage2
- Qwen/Qwen3-4B
- Qwen/Qwen3-ASR-1.7B
license: cc-by-nc-4.0
short_description: Generate songs and covers from style prompts and lyrics using YuE2-3B.
---

# 🎵 YuE2-3B Music Generator & Studio

This Hugging Face Space uses the **YuE2-3B** open music foundation model to generate full-length songs and AI covers from musical style descriptions and section-labeled lyrics.

## 🚀 Features & Models
- **Music Generation**: Powered by [`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B) with full vocal and instrumental arrangement.
- **Audio Decoder**: High fidelity audio rendering via [`m-a-p/YuE2-Vae`](https://huggingface.co/m-a-p/YuE2-Vae).
- **Melody & Score Transcription**: [`m-a-p/SheetSage2`](https://huggingface.co/m-a-p/SheetSage2) extracts melody into editable ABC notation.
- **AI Lyricist**: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) drafts structured lyrics in various languages.
- **Vocal Transcription (ASR)**: [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) transcribes lyrics from source audio for cover creation.

## 🎨 UI Customization
The web interface appearance can be customized in [`style.css`](style.css) and [`app.py`](app.py).
See [`UI_CUSTOMIZATION.md`](UI_CUSTOMIZATION.md) for a guide on styling variables, colors, layouts, and components.
