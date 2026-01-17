# ShortsFactory AI

An automated tool to create viral shorts from long-form videos using local AI.

## Repository
[https://github.com/flamacore/ShortsFactoryAI](https://github.com/flamacore/ShortsFactoryAI)

## Features
- **GPU Accelerated**: Uses CUDA for Whisper, Vision, Face Detection, and Rendering (NVENC).
- **Smart Director**: Llama 3 (or other LLMs) analyzes the video timeline to pick the best parts.
- **Auto-Cropping**: YOLOv8-Face tracking ensures the speaker stays in the frame.
- **Dynamic Subtitles**: Auto-generated subtitles burned into the video.
- **Privacy First**: Everything runs locally using Ollama and Faster-Whisper.

## Requirements
- NVIDIA GPU with CUDA support.
- [Ollama](https://ollama.com/) installed and running.
- Python 3.10+
- FFmpeg installed and added to PATH.

## Setup
1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Pull Ollama Models**:
   Ensure you have a vision model and a text model.
   ```bash
   ollama pull llava
   ollama pull llama3
   ```

## Running
Run the usage script:
```powershell
./run.ps1
```
Or manually:
```bash
streamlit run app.py
```

## Configuration
Edit `config.yaml` to tweak settings like resolution, chunk size, or model defaults.
