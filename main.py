import os
import argparse
import json
import torch
import gc
from typing import List
from src.media_processor import MediaProcessor
from src.transcriber import AudioTranscriber
from src.vision_analyzer import VisionAnalyzer
from src.models import VideoMetadata
from src.indexer import VideoIndexer
import concurrent.futures

def cleanup_gpu():
    """Aggressively clean up GPU memory (mostly for Whisper/Torch)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def process_batch(video_paths: List[str], args):
    processor = MediaProcessor(output_root=args.output_dir)
    videos_metadata: List[VideoMetadata] = []

    # --- Stage 1: Media Processing (FFmpeg) ---
    print(f"\n>>> STAGE 1: Media Processing ({len(video_paths)} videos)")
    for path in video_paths:
        print(f"Processing: {os.path.basename(path)}")
        meta = processor.prepare_video_metadata(path)
        meta.audio_path = processor.extract_audio(path)
        # Use adaptive extraction
        meta.frames = processor.extract_frames(path, max_frames=args.max_frames)
        videos_metadata.append(meta)
        print(f"  Done. Extracted {len(meta.frames)} frames.")
    
    cleanup_gpu()

    # --- Stage 2: Transcription (Whisper) ---
    print(f"\n>>> STAGE 2: Audio Transcription")
    try:
        transcriber = AudioTranscriber(model_size=args.whisper_model)
        for meta in videos_metadata:
            if meta.audio_path:
                print(f"Transcribing audio for: {os.path.basename(meta.video_path)}")
                meta.transcription = transcriber.transcribe(meta.audio_path)
                
                # Save intermediate transcription to file
                trans_file = os.path.join(meta.base_dir, "transcription.txt")
                with open(trans_file, "w", encoding="utf-8") as f:
                    f.write(meta.transcription)
                print(f"  Saved transcription to: {trans_file}")
    finally:
        if 'transcriber' in locals():
            del transcriber
        cleanup_gpu()

    # --- Stage 3: Vision Analysis (GGUF via llama-server) ---
    print(f"\n>>> STAGE 3: Visual Analysis & Summarization (PARALLEL MODE)")
    model_path = args.vlm_model
    if not model_path.endswith(".gguf"):
        # model_path = "models/Qwen3.5-9B-Q8_0.gguf"
        model_path = "models/Qwen3.5-9B-UD-Q4_K_XL.gguf"
            
    # Допоміжна функція для обробки одного відео у потоці
    def process_vision_task(meta, analyzer_instance):
        try:
            print(f"[{os.path.basename(meta.video_path)}] Analyzing frames...")
            context_text = meta.transcription if meta.transcription else "No audio context available."
            
            # Аналіз кадрів
            meta.frames = analyzer_instance.analyze_frames(meta.frames, context_text, batch_size=args.batch_size)
            
            print(f"[{os.path.basename(meta.video_path)}] Generating summary...")
            # Генерація підсумку
            meta.summary = analyzer_instance.generate_video_summary(meta.frames, context_text)
            
            # Примусове очищення слота для цього конкретного потоку (якщо використовуємо старий метод)
            # Але з новим LLAMA_SET_ROWS=1 сервер сам краще керує пам'яттю слотів.
            
            print(f"[{os.path.basename(meta.video_path)}] DONE!")
        except Exception as task_e:
            print(f"[{os.path.basename(meta.video_path)}] Error during vision task: {task_e}")

    try:
        with VisionAnalyzer(model_id=model_path) as analyzer:
            # Використовуємо ThreadPoolExecutor для паралельних запитів.
            # max_workers має дорівнювати параметру --parallel у сервері!
            max_parallel_requests = 4 
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
                # Запускаємо задачі для всіх відео
                futures = [executor.submit(process_vision_task, meta, analyzer) for meta in videos_metadata]
                
                # Чекаємо завершення всіх потоків
                concurrent.futures.wait(futures)

    except Exception as e:
        print(f"Critical error during vision analysis: {e}")
    finally:
        cleanup_gpu()

    # --- Stage 4: Saving Results ---
    print(f"\n>>> STAGE 4: Saving Results")
    for meta in videos_metadata:
        video_name = os.path.basename(meta.base_dir)
        result_path = os.path.join(meta.base_dir, "analysis_result.json")
        
        result_data = {
            "video_path": meta.video_path,
            "duration": meta.duration,
            "transcription": meta.transcription,
            "summary": getattr(meta, "summary", "Summary generation failed."),
            "analysis": [
                {"timestamp": f.timestamp, "description": f.description, "frame_path": f.path} 
                for f in meta.frames
            ]
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"Results for {video_name} saved to {result_path}")

    # --- Stage 5: Vector Indexing ---
    print(f"\n>>> STAGE 5: Vector Indexing")
    try:
        indexer = VideoIndexer()
        for meta in videos_metadata:
            result_path = os.path.join(meta.base_dir, "analysis_result.json")
            if os.path.exists(result_path):
                indexer.index_json(result_path)
    except Exception as e:
        print(f"Error during vector indexing: {e}")
    finally:
        if 'indexer' in locals():
            del indexer
        cleanup_gpu()

def main():
    parser = argparse.ArgumentParser(description="Batch Multimodal Video Analysis System")
    parser.add_argument("--input", type=str, required=True, help="Path to video file or directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Root directory for processed data")

    parser.add_argument("--vlm_model", type=str, default="models/Qwen3.5-9B-UD-Q4_K_XL.gguf", help="Path to GGUF model")
    #parser.add_argument("--vlm_model", type=str, default="models/Qwen3.5-9B-Q8_0.gguf", help="Path to GGUF model")


    parser.add_argument("--whisper_model", type=str, default="large", help="Whisper model size")
    parser.add_argument("--threshold", type=float, default=0.1, help="Deprecated (adaptive)")
    parser.add_argument("--interval", type=float, default=30.0, help="Deprecated (adaptive)")
    parser.add_argument("--max_frames", type=int, default=50, help="Target max frames per video (adaptive)")
    parser.add_argument("--batch_size", type=int, default=0, help="Number of frames to analyze at once (0 = auto)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for vision analysis")
    
    args = parser.parse_args()

    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    video_paths = []

    if os.path.isfile(args.input):
        video_paths.append(args.input)
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))
    else:
        print(f"Error: Input {args.input} not found.")
        return

    if not video_paths:
        print("No video files found to process.")
        return

    print(f"Found {len(video_paths)} video(s) to process.")
    process_batch(video_paths, args)

if __name__ == "__main__":
    main()
