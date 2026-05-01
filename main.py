import os
import argparse
import json
import re
import shutil
import time
import torch
import gc
from pathlib import Path
from typing import List, Optional
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

def estimate_vlm_cost(meta: VideoMetadata) -> int:
    frames_for_request = meta.frames[:50]
    frames_bytes = 0
    for frame in frames_for_request:
        try:
            frames_bytes += os.path.getsize(frame.path)
        except OSError:
            pass

    transcription_chars = len(meta.transcription or "")
    estimated_cost = frames_bytes + transcription_chars * 4

    meta.timings["vlm_frames_total"] = len(meta.frames)
    meta.timings["vlm_frames_sent"] = len(frames_for_request)
    meta.timings["vlm_frames_bytes"] = frames_bytes
    meta.timings["vlm_transcription_chars"] = transcription_chars
    meta.timings["vlm_estimated_cost"] = estimated_cost
    return estimated_cost

def record_transcription_config(meta: VideoMetadata, config: dict):
    meta.timings["transcription_config"] = config
    meta.timings["transcription_backend"] = config.get("backend")
    meta.timings["transcription_model"] = config.get("model")
    meta.timings["transcription_beam_size"] = config.get("beam_size")
    meta.timings["transcription_batched"] = config.get("batched")
    meta.timings["transcription_batch_size"] = config.get("batch_size")
    meta.timings["transcription_vad_filter"] = config.get("vad_filter")

def sanitized_video_name(video_path: str) -> str:
    return re.sub(r'[^\w\-]', '_', Path(video_path).stem)

def analysis_result_path(output_dir: str, video_path: str) -> str:
    return os.path.join(output_dir, sanitized_video_name(video_path), "analysis_result.json")

def is_valid_analysis_result(result_path: str) -> bool:
    if not os.path.exists(result_path):
        return False

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    summary = str(data.get("summary") or "").strip()
    segments = data.get("segments") or []
    failed_summaries = {
        "",
        "Structured analysis failed.",
        "Summary generation failed.",
        "No video content analyzed.",
    }
    return summary not in failed_summaries and bool(segments)

def build_analysis_result(meta: VideoMetadata) -> dict:
    segments_data = [
        {
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "description": segment.description
        }
        for segment in meta.segments
    ]

    legacy_analysis = []
    if meta.segments:
        for segment in meta.segments:
            frame_index = max(0, min(segment.start_frame, len(meta.frames) - 1)) if meta.frames else 0
            frame_path = meta.frames[frame_index].path if meta.frames else ""
            legacy_analysis.append({
                "timestamp": segment.start_time,
                "description": segment.description,
                "frame_path": frame_path
            })
    else:
        legacy_analysis = [
            {"timestamp": f.timestamp, "description": f.description, "frame_path": f.path}
            for f in meta.frames
        ]

    return {
        "video_path": meta.video_path,
        "duration": meta.duration,
        "transcription": meta.transcription,
        "transcription_config": meta.timings.get("transcription_config", {}),
        "summary": meta.summary or "Summary generation failed.",
        "segments": segments_data,
        "analysis": legacy_analysis,
        "timings": meta.timings
    }

def write_analysis_result(meta: VideoMetadata) -> str:
    result_path = os.path.join(meta.base_dir, "analysis_result.json")
    temp_path = f"{result_path}.tmp"
    result_data = build_analysis_result(meta)

    os.makedirs(meta.base_dir, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    os.replace(temp_path, result_path)
    return result_path

def completed_video_path(original_path: str, args) -> Path:
    source = Path(original_path)
    completed_root = Path(args.completed_dir)
    input_path = Path(args.input)

    if input_path.is_dir():
        try:
            relative_path = source.resolve().relative_to(input_path.resolve())
        except ValueError:
            relative_path = Path(source.name)
    else:
        relative_path = Path(source.name)

    destination = completed_root / relative_path
    if not destination.exists() or source.resolve() == destination.resolve():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}_{counter}{destination.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1

def copy_video_to_completed(original_path: str, args) -> str:
    if not args.archive_completed_videos:
        return original_path

    source = Path(original_path)
    if not source.exists():
        raise FileNotFoundError(f"Original video file not found: {source}")

    destination = completed_video_path(original_path, args)
    if source.resolve() == destination.resolve():
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)

def remove_original_video(original_path: str, archived_path: str, args) -> None:
    if not args.archive_completed_videos:
        return

    source = Path(original_path)
    destination = Path(archived_path)
    if source.exists() and source.resolve() != destination.resolve():
        source.unlink()

def is_successful_vlm_result(meta: VideoMetadata) -> bool:
    summary = str(meta.summary or "").strip()
    failed_summaries = {
        "",
        "Structured analysis failed.",
        "Summary generation failed.",
        "No video content analyzed.",
    }
    return summary not in failed_summaries and bool(meta.segments)

def finalize_successful_video(meta: VideoMetadata, args) -> str:
    original_path = meta.video_path
    archived_path = copy_video_to_completed(original_path, args)
    meta.timings["source_video_path"] = original_path
    meta.timings["archived_video_path"] = archived_path
    meta.video_path = archived_path

    result_path = write_analysis_result(meta)
    remove_original_video(original_path, archived_path, args)
    return result_path

def finalize_existing_analysis(video_path: str, args) -> str:
    result_path = analysis_result_path(args.output_dir, video_path)
    if not args.archive_completed_videos:
        return result_path

    archived_path = copy_video_to_completed(video_path, args)
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    result_data["video_path"] = archived_path
    timings = result_data.setdefault("timings", {})
    timings["source_video_path"] = video_path
    timings["archived_video_path"] = archived_path

    temp_path = f"{result_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    os.replace(temp_path, result_path)

    remove_original_video(video_path, archived_path, args)
    return result_path

def index_analysis_results(result_paths: List[str], args) -> None:
    unique_result_paths = list(dict.fromkeys(result_paths))
    if not unique_result_paths:
        print("No analysis results to index.")
        return

    print(f"\n>>> STAGE 5: Vector Indexing ({len(unique_result_paths)} result file(s))")
    try:
        indexer = VideoIndexer(
            embedding_backend=args.embedding_backend,
            table_name=args.embedding_table,
            embedding_context_size=args.embedding_ctx_size,
            embedding_parallel=args.embedding_parallel,
        )
        for result_path in unique_result_paths:
            if os.path.exists(result_path):
                indexer.index_json(result_path)
    except Exception as e:
        print(f"Error during vector indexing: {e}")
    finally:
        if 'indexer' in locals():
            indexer.close()
            del indexer
        cleanup_gpu()

def process_batch(video_paths: List[str], args, existing_result_paths: Optional[List[str]] = None):
    pipeline_start = time.perf_counter()
    processor = MediaProcessor(output_root=args.output_dir)
    videos_metadata: List[VideoMetadata] = []
    result_paths_to_index: List[str] = list(existing_result_paths or [])
    completed_metas: List[VideoMetadata] = []
    failed_metas: List[VideoMetadata] = []

    # --- Stage 1: Media Processing (FFmpeg) ---
    print(f"\n>>> STAGE 1: Media Processing ({len(video_paths)} videos)")
    stage1_start = time.perf_counter()
    for path in video_paths:
        video_stage1_start = time.perf_counter()
        print(f"Processing: {os.path.basename(path)}")
        meta = processor.prepare_video_metadata(path)
        meta.audio_path = processor.extract_audio(path)
        # Use adaptive extraction
        meta.frames = processor.extract_frames(path, max_frames=args.max_frames)
        meta.timings["media_processing_seconds"] = round(time.perf_counter() - video_stage1_start, 3)
        videos_metadata.append(meta)
        print(f"  Done. Extracted {len(meta.frames)} frames.")
    stage1_elapsed = time.perf_counter() - stage1_start
    print(f">>> STAGE 1 completed in {stage1_elapsed:.2f}s")
    
    cleanup_gpu()

    # --- Stage 2: Transcription (Whisper) ---
    print(f"\n>>> STAGE 2: Audio Transcription")
    stage2_start = time.perf_counter()
    try:
        transcriber = AudioTranscriber(
            model_size=args.whisper_model,
            backend=args.whisper_backend,
            beam_size=args.whisper_beam_size,
            batch_size=args.whisper_batch_size,
            batched=args.whisper_batched,
            vad_filter=args.whisper_vad_filter,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            cuda_dll_dir=args.cuda_dll_dir,
        )
        transcription_config = transcriber.config
        for meta in videos_metadata:
            record_transcription_config(meta, transcription_config)
            if meta.audio_path:
                transcription_start = time.perf_counter()
                print(f"Transcribing audio for: {os.path.basename(meta.video_path)}")
                meta.transcription = transcriber.transcribe(meta.audio_path)
                meta.timings["transcription_seconds"] = round(time.perf_counter() - transcription_start, 3)
                
                # Save intermediate transcription to file
                trans_file = os.path.join(meta.base_dir, "transcription.txt")
                with open(trans_file, "w", encoding="utf-8") as f:
                    f.write(meta.transcription)
                print(f"  Saved transcription to: {trans_file}")
        stage2_elapsed = time.perf_counter() - stage2_start
        print(f">>> STAGE 2 completed in {stage2_elapsed:.2f}s")
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

    stage3_start = time.perf_counter()
    stage3_elapsed = 0.0
    stage3_interrupted = False
    max_parallel_requests = max(1, args.parallel_requests)
    server_parallel_slots = max(1, args.server_parallel)

    for meta in videos_metadata:
        estimate_vlm_cost(meta)

    scheduled_videos = sorted(
        videos_metadata,
        key=lambda item: item.timings.get("vlm_estimated_cost", 0),
        reverse=True,
    )
    print("\n>>> VLM schedule order (estimated heaviest first)")
    for rank, meta in enumerate(scheduled_videos, 1):
        meta.timings["vlm_schedule_rank"] = rank
        frames_mb = meta.timings.get("vlm_frames_bytes", 0) / (1024 * 1024)
        print(
            f"  {rank:02d}. {os.path.basename(meta.video_path)} | "
            f"frames={meta.timings.get('vlm_frames_sent', 0)}/{meta.timings.get('vlm_frames_total', 0)} | "
            f"frames_mb={frames_mb:.2f} | "
            f"transcription_chars={meta.timings.get('vlm_transcription_chars', 0)} | "
            f"cost={meta.timings.get('vlm_estimated_cost', 0)}"
        )
            
    # Допоміжна функція для обробки одного відео у потоці
    def process_vision_task(meta, analyzer_instance):
        video_name = os.path.basename(meta.video_path)
        vlm_start = time.perf_counter()
        try:
            print(f"[{video_name}] Analyzing frames...")
            context_text = meta.transcription if meta.transcription else "No audio context available."
            
            # Аналіз кадрів
            meta.summary, meta.segments, vlm_timings = analyzer_instance.analyze_video_structured(
                meta.frames,
                context_text,
                batch_size=args.batch_size
            )
            
            # Генерація підсумку
            vlm_elapsed = time.perf_counter() - vlm_start
            meta.timings.update(vlm_timings)
            meta.timings["vlm_task_wall_seconds"] = round(vlm_elapsed, 3)
            meta.timings["vlm_segments_count"] = len(meta.segments)
            meta.timings["vlm_success"] = is_successful_vlm_result(meta)

            if not meta.timings["vlm_success"]:
                raise RuntimeError(f"VLM returned an unsuccessful result for {video_name}: {meta.summary}")

            server_total = meta.timings.get("vlm_server_total_seconds")
            if server_total is not None:
                print(
                    f"[{video_name}] Structured summary and segments generated: "
                    f"{server_total:.2f}s server active / {vlm_elapsed:.2f}s client wall."
                )
            else:
                print(f"[{video_name}] Structured summary and segments generated in {vlm_elapsed:.2f}s client wall.")
            
            # Примусове очищення слота для цього конкретного потоку (якщо використовуємо старий метод)
            # Але з новим LLAMA_SET_ROWS=1 сервер сам краще керує пам'яттю слотів.
            
            print(f"[{video_name}] DONE!")
        except Exception as task_e:
            meta.timings["vlm_task_wall_seconds"] = round(time.perf_counter() - vlm_start, 3)
            meta.timings["vlm_success"] = False
            meta.timings["vlm_error"] = str(task_e)
            print(f"[{video_name}] Error during vision task: {task_e}")
            raise

    try:
        with VisionAnalyzer(model_id=model_path, verbose=args.verbose, parallel_slots=server_parallel_slots) as analyzer:
            # Використовуємо ThreadPoolExecutor для паралельних запитів.
            # max_workers має дорівнювати параметру --parallel у сервері!
            print(f">>> VLM timing note: per-video wall time includes HTTP wait, prompt processing, generation, and any server-slot queueing.")
            print(f">>> Executor workers: {max_parallel_requests}; llama-server --parallel: {server_parallel_slots}")

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests)
            finalizer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            active_futures = {}
            finalizer_futures = {}
            pending_videos = iter(scheduled_videos)

            def submit_next_video() -> bool:
                try:
                    next_meta = next(pending_videos)
                except StopIteration:
                    return False
                future = executor.submit(process_vision_task, next_meta, analyzer)
                active_futures[future] = next_meta
                return True

            def submit_finalizer(meta) -> None:
                video_name = os.path.basename(meta.video_path)
                future = finalizer_executor.submit(finalize_successful_video, meta, args)
                finalizer_futures[future] = meta
                print(f"[{video_name}] Queued async archive/save finalizer.")

            def finalize_archive_future(future, meta) -> None:
                video_name = os.path.basename(meta.timings.get("source_video_path") or meta.video_path)
                try:
                    result_path = future.result()
                    result_paths_to_index.append(result_path)
                    completed_metas.append(meta)
                    print(f"[{video_name}] Result saved and source archived: {result_path}")
                except Exception as finalize_e:
                    meta.timings["archive_success"] = False
                    meta.timings["archive_error"] = str(finalize_e)
                    failed_metas.append(meta)
                    print(f"[{video_name}] VLM succeeded, but archive/save failed; left in input for retry: {finalize_e}")

            def collect_finished_finalizers() -> None:
                for future, meta in list(finalizer_futures.items()):
                    if future.done():
                        finalizer_futures.pop(future, None)
                        finalize_archive_future(future, meta)

            def wait_for_finalizers() -> None:
                while finalizer_futures:
                    done_futures, _ = concurrent.futures.wait(
                        finalizer_futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        meta = finalizer_futures.pop(future)
                        finalize_archive_future(future, meta)

            def finalize_vision_future(future, meta) -> None:
                video_name = os.path.basename(meta.video_path)
                try:
                    future.result()
                    submit_finalizer(meta)
                except concurrent.futures.CancelledError:
                    meta.timings["vlm_success"] = False
                    meta.timings["vlm_error"] = "Cancelled by user"
                    failed_metas.append(meta)
                    print(f"[{video_name}] Cancelled and left in input for retry.")
                except Exception as task_e:
                    failed_metas.append(meta)
                    print(f"[{video_name}] Left in input for retry: {task_e}")

            def finalize_already_done_futures() -> None:
                for future, meta in list(active_futures.items()):
                    if future.done():
                        active_futures.pop(future, None)
                        finalize_vision_future(future, meta)

            try:
                # Тримаємо в executor тільки активні VLM-запити. Наступне відео
                # стартує після завершення попереднього VLM-запиту, а copy/json/delete
                # виконуються окремим finalizer executor-ом.
                for _ in range(max_parallel_requests):
                    if not submit_next_video():
                        break

                # Успішний VLM-result одразу передаємо в async finalizer.
                while active_futures:
                    done_futures, _ = concurrent.futures.wait(
                        active_futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        meta = active_futures.pop(future)
                        finalize_vision_future(future, meta)
                        collect_finished_finalizers()
                        submit_next_video()
                wait_for_finalizers()
            except KeyboardInterrupt:
                stage3_interrupted = True
                print("\n>>> Ctrl+C received. Finalizing completed VLM tasks, cancelling active tasks, and stopping llama-server...")
                finalize_already_done_futures()
                collect_finished_finalizers()
                for future, meta in active_futures.items():
                    if not future.done():
                        future.cancel()
                        meta.timings["vlm_success"] = False
                        meta.timings["vlm_error"] = "Interrupted by user"
                executor.shutdown(wait=False, cancel_futures=True)
                analyzer._stop_server()
                done_after_stop, _ = concurrent.futures.wait(
                    active_futures,
                    timeout=5,
                    return_when=concurrent.futures.ALL_COMPLETED,
                )
                for future in done_after_stop:
                    meta = active_futures.pop(future, None)
                    if meta is not None:
                        finalize_vision_future(future, meta)
                finalize_already_done_futures()
                print(">>> Waiting for already completed videos to finish archive/save...")
                wait_for_finalizers()
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                finalizer_executor.shutdown(wait=True, cancel_futures=False)
                collect_finished_finalizers()

    except KeyboardInterrupt:
        print(">>> VLM analysis interrupted by user. Completed videos already saved; unfinished videos remain in input for retry.")
    except Exception as e:
        print(f"Critical error during vision analysis: {e}")
    finally:
        stage3_elapsed = time.perf_counter() - stage3_start
        for meta in videos_metadata:
            meta.timings["stage3_total_wall_seconds"] = round(stage3_elapsed, 3)
            meta.timings["stage3_executor_workers"] = max_parallel_requests
            meta.timings["stage3_server_parallel_slots"] = server_parallel_slots
        print(f">>> STAGE 3 completed in {stage3_elapsed:.2f}s wall time for {len(videos_metadata)} video(s)")
        cleanup_gpu()

    if stage3_interrupted:
        for meta in completed_metas:
            result_path = write_analysis_result(meta)
            print(f"Final timings saved to {result_path}")
        print(">>> Pipeline stopped after Ctrl+C.")
        return

    # --- Stage 4: Saving Results ---
    print(f"\n>>> STAGE 4: Finalizing Saved Results")
    for meta in completed_metas:
        result_path = write_analysis_result(meta)
        print(f"Final timings saved to {result_path}")

    if failed_metas:
        print(f">>> {len(failed_metas)} video(s) failed VLM analysis and remain in input for retry.")
    print(f">>> {len(completed_metas)} video(s) completed and archived.")

    # --- Stage 5: Vector Indexing ---
    index_analysis_results(result_paths_to_index, args)

    pipeline_elapsed = time.perf_counter() - pipeline_start
    print(f"\n>>> TOTAL PIPELINE completed in {pipeline_elapsed:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Batch Multimodal Video Analysis System")
    parser.add_argument("--input", type=str, required=True, help="Path to video file or directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Root directory for processed data")
    parser.add_argument("--completed_dir", type=str, default=os.path.join("data", "completed_videos"), help="Directory for successfully analyzed source videos")
    parser.add_argument("--archive_completed_videos", action=argparse.BooleanOptionalAction, default=True, help="Copy successful videos to completed_dir and remove them from input")

    parser.add_argument("--vlm_model", type=str, default="models/Qwen3.5-9B-UD-Q4_K_XL.gguf", help="Path to GGUF model")
    #parser.add_argument("--vlm_model", type=str, default="models/Qwen3.5-9B-Q8_0.gguf", help="Path to GGUF model")

    parser.add_argument("--whisper_backend", choices=["faster", "openai"], default="faster", help="Whisper backend")
    parser.add_argument("--whisper_model", type=str, default="turbo", help="Whisper model size/name")
    parser.add_argument("--whisper_beam_size", type=int, default=3, help="Whisper beam size")
    parser.add_argument("--whisper_batch_size", type=int, default=16, help="faster-whisper batched pipeline batch size")
    parser.add_argument("--whisper_batched", action=argparse.BooleanOptionalAction, default=True, help="Enable faster-whisper batched pipeline")
    parser.add_argument("--whisper_vad_filter", action=argparse.BooleanOptionalAction, default=False, help="Enable faster-whisper VAD filter")
    parser.add_argument("--whisper_compute_type", type=str, default="float16", help="faster-whisper compute type")
    parser.add_argument("--whisper_device", type=str, default="cuda", help="faster-whisper device")
    parser.add_argument("--cuda_dll_dir", type=str, default=os.path.join("bin", "llama-cpp"), help="Directory with CUDA DLLs for faster-whisper on Windows")
    parser.add_argument("--threshold", type=float, default=0.1, help="Deprecated (adaptive)")
    parser.add_argument("--interval", type=float, default=30.0, help="Deprecated (adaptive)")
    parser.add_argument("--max_frames", type=int, default=50, help="Target max frames per video (adaptive)")
    parser.add_argument("--batch_size", type=int, default=0, help="Number of frames to analyze at once (0 = auto)")
    parser.add_argument("--parallel_requests", type=int, default=2, help="Client-side VLM request concurrency")
    parser.add_argument("--server_parallel", type=int, default=2, help="llama-server --parallel slots")
    parser.add_argument("--embedding_backend", choices=["minilm", "qwen3-gguf", "qwen3-4b-gguf"], default=None, help="Embedding backend for Stage 5 indexing")
    parser.add_argument("--embedding_table", default=None, help="Override LanceDB table for Stage 5 indexing")
    parser.add_argument("--embedding_ctx_size", type=int, default=None, help="llama-server context size for Stage 5 embedding indexing")
    parser.add_argument("--embedding_parallel", type=int, default=2, help="llama-server --parallel slots for Stage 5 embedding indexing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for vision analysis")
    
    args = parser.parse_args()

    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    discovered_video_paths = []

    if os.path.isfile(args.input):
        discovered_video_paths.append(args.input)
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(video_extensions):
                    discovered_video_paths.append(os.path.join(root, file))
    else:
        print(f"Error: Input {args.input} not found.")
        return

    if not discovered_video_paths:
        print("No video files found to process.")
        return

    video_paths = []
    existing_result_paths = []
    for path in discovered_video_paths:
        result_path = analysis_result_path(args.output_dir, path)
        if is_valid_analysis_result(result_path):
            print(f"Skipping already analyzed video: {os.path.basename(path)}")
            try:
                existing_result_paths.append(finalize_existing_analysis(path, args))
            except Exception as e:
                print(f"  -> Warning: could not archive existing analyzed video {path}: {e}")
                existing_result_paths.append(result_path)
            continue
        video_paths.append(path)

    if not video_paths:
        print("No new or failed videos to process.")
        if existing_result_paths:
            index_analysis_results(existing_result_paths, args)
        return

    print(f"Found {len(video_paths)} video(s) to process.")
    if existing_result_paths:
        print(f"Found {len(existing_result_paths)} already analyzed video(s) to index/confirm.")
    process_batch(video_paths, args, existing_result_paths=existing_result_paths)

if __name__ == "__main__":
    main()
