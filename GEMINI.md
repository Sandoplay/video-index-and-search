# Multimodal Video Indexer & Search

This project is a high-performance, local AI system designed to analyze videos by extracting visual and audio information. It generates detailed textual descriptions and summaries, enabling semantic search and indexing of video content.

## Project Overview

The system processes video files to create structured metadata (JSON) containing:
1.  **Audio Transcription:** Speech-to-text using OpenAI Whisper.
2.  **Visual Analysis:** Chronological descriptions of video frames using the Qwen3-VL Vision-Language Model (VLM).
3.  **Summary:** A cohesive summary of the entire video content.

It is highly optimized for NVIDIA GPUs (specifically RTX 50-series Blackwell architecture) using CUDA-accelerated inference.

### Key Technologies

*   **Language:** Python 3.13
*   **Media Processing:** FFmpeg (adaptive frame extraction).
*   **Audio:** OpenAI Whisper (PyTorch).
*   **Vision AI:** `llama.cpp` (server mode) running `Qwen3-VL-8B-Instruct` (Unsloth GGUF format) with `mmproj` vision encoder.
*   **Inference:** CUDA 13.0+ optimized binaries for maximum throughput (~100-150 tokens/s generation).

## Architecture

The project is structured into modular components within `src/`:

*   **`src/media_processor.py`**: Handles video file operations.
    *   Extracts audio to 16kHz mono WAV.
    *   Performs **adaptive frame extraction**: Calculates optimal frame rate based on video duration to extract a target number of frames (default ~50), preventing data overload.
*   **`src/transcriber.py`**: Wraps the OpenAI Whisper model for audio transcription.
*   **`src/vision_analyzer.py`**: A robust client for the local `llama-server`.
    *   Manages the `llama-server.exe` subprocess.
    *   Handles **auto-batching**: Dynamically calculates how many images can fit into the context window (32k tokens) to process multiple frames in a single API call.
    *   Generates global summaries.
*   **`src/models.py`**: Data classes (`VideoMetadata`, `FrameMetadata`) for structured data passing.
*   **`main.py`**: The CLI entry point that orchestrates the pipeline (Media -> Audio -> Vision -> Save).

## Setup & Prerequisites

### 1. Environment
Ensure a Python 3.12+ environment is active.
```bash
# Install python dependencies
pip install -r requirements.txt
```
*Note: `llama-cpp-python` is not strictly required for the core pipeline anymore as it uses the standalone `llama-server.exe`, but `requests`, `torch`, `openai-whisper` are essential.*

### 2. Binaries (`bin/`)
The project relies on local binaries to ensure compatibility and performance:
*   `bin/ffmpeg.exe` & `bin/ffprobe.exe`: For media processing.
*   `bin/llama-cpp/llama-server.exe`: The inference engine.

### 3. Models (`models/`)
Required GGUF files for the vision pipeline:
*   `models/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf`: The main language model (Unsloth Quantized).
*   `models/mmproj-Qwen3VL-8B-Instruct-F16.gguf`: The vision encoder (Projector) for Qwen3-VL.

## Running the System

To process a single video or a directory of videos:

```powershell
python main.py --input "path/to/video.mp4"
```

### Optional Arguments
*   `--output_dir`: Directory to save results (default: `data/processed`).
*   `--max_frames`: Target number of frames to extract (default: `50`).
*   `--batch_size`: Number of frames to analyze in one prompt. Set to `0` for **auto-calculation** based on VRAM/Context (Recommended).
*   `--whisper_model`: Whisper model size (default: `base`).

### Example
```powershell
# Process all videos in a folder, limiting to 40 frames each, using auto-batching
python main.py --input "data/input" --max_frames 40 --batch_size 0
```

## Output Structure

Results are saved in `data/processed/<video_name>/analysis_result.json`:

```json
{
  "video_path": "...",
  "duration": 161.25,
  "transcription": "Full audio text...",
  "summary": "A detailed paragraph summarizing the whole video...",
  "analysis": [
    {
      "timestamp": 0.0,
      "description": "Visual description of this segment...",
      "frame_path": "..."
    },
    ...
  ]
}
```

## Development Conventions

*   **Dependency Isolation:** All python packages are in `.venv`.
*   **Local Binaries:** Prefer using binaries in `bin/` over system PATH to ensure version consistency (especially for FFmpeg and Llama.cpp).
*   **Error Handling:** The `VisionAnalyzer` implements robust server health checks and retries to handle the asynchronous nature of the `llama-server` startup.
*   **Performance:** The pipeline is designed for batch processing. `cleanup_gpu()` is called between stages to manage VRAM usage on consumer cards (16GB recommended for parallel execution).
