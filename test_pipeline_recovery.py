import argparse
import json
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import main as pipeline
from src.models import FrameMetadata, SegmentMetadata, VideoMetadata


TEST_ROOT = Path("data/temp_pipeline_recovery_tests")


class FakeMediaProcessor:
    fail_prepare = set()
    fail_audio = set()
    fail_frames = set()

    def __init__(self, output_root: str):
        self.output_root = Path(output_root)

    def _base_dir(self, video_path: str) -> Path:
        return self.output_root / pipeline.sanitized_video_name(video_path)

    def prepare_video_metadata(self, video_path: str) -> VideoMetadata:
        if Path(video_path).stem in type(self).fail_prepare:
            raise RuntimeError("simulated metadata failure")

        base_dir = self._base_dir(video_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        return VideoMetadata(
            video_path=video_path,
            base_dir=str(base_dir),
            duration=1.0,
        )

    def extract_audio(self, video_path: str) -> str:
        if Path(video_path).stem in type(self).fail_audio:
            raise RuntimeError("simulated audio extraction failure")

        audio_path = self._base_dir(video_path) / "audio.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake audio")
        return str(audio_path)

    def extract_frames(self, video_path: str, max_frames: int = 40):
        if Path(video_path).stem in type(self).fail_frames:
            raise RuntimeError("simulated frame extraction failure")

        frames_dir = self._base_dir(video_path) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame_0001.jpg"
        frame_path.write_bytes(b"fake frame")
        return [FrameMetadata(path=str(frame_path), timestamp=0.0, index=0)]


class FakeAudioTranscriber:
    fail_for = set()

    config = {
        "backend": "fake",
        "model": "fake",
        "beam_size": 1,
        "batched": False,
        "batch_size": 1,
        "vad_filter": False,
    }

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path: str) -> str:
        video_key = Path(audio_path).parents[0].name
        if video_key in type(self).fail_for:
            raise RuntimeError("simulated transcription failure")
        return "fake transcription"


class FakeVideoIndexer:
    indexed_paths = []

    def __init__(self, *args, **kwargs):
        pass

    def index_json(self, result_path: str) -> None:
        self.indexed_paths.append(result_path)

    def close(self) -> None:
        pass


class FakeVisionAnalyzer:
    behaviors = {}
    delays = {}
    starts = {}
    started_events = {}
    release_events = {}
    stopped = False

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_server()

    def _stop_server(self):
        type(self).stopped = True

    def analyze_video_structured(self, frames, transcription, batch_size=0):
        video_key = Path(frames[0].path).parents[1].name
        type(self).starts[video_key] = time.perf_counter()
        if video_key in type(self).started_events:
            type(self).started_events[video_key].set()
        if video_key in type(self).release_events:
            type(self).release_events[video_key].wait(timeout=10)

        delay = type(self).delays.get(video_key, 0.0)
        if delay:
            time.sleep(delay)

        behavior = type(self).behaviors.get(video_key, "success")
        if behavior == "failure":
            raise RuntimeError("simulated VLM failure")
        if behavior == "failed_result":
            return "Structured analysis failed.", [], {"vlm_client_wall_seconds": delay}
        if behavior == "keyboard_interrupt":
            raise KeyboardInterrupt()

        segment = SegmentMetadata(
            start_frame=0,
            end_frame=0,
            start_time=0.0,
            end_time=1.0,
            description=f"valid segment for {video_key}",
        )
        return f"valid summary for {video_key}", [segment], {"vlm_client_wall_seconds": delay}


@contextmanager
def patched_pipeline():
    original_media_processor = pipeline.MediaProcessor
    original_transcriber = pipeline.AudioTranscriber
    original_vision_analyzer = pipeline.VisionAnalyzer
    original_indexer = pipeline.VideoIndexer
    original_cleanup_gpu = pipeline.cleanup_gpu

    FakeVideoIndexer.indexed_paths = []
    FakeMediaProcessor.fail_prepare = set()
    FakeMediaProcessor.fail_audio = set()
    FakeMediaProcessor.fail_frames = set()
    FakeAudioTranscriber.fail_for = set()
    FakeVisionAnalyzer.behaviors = {}
    FakeVisionAnalyzer.delays = {}
    FakeVisionAnalyzer.starts = {}
    FakeVisionAnalyzer.started_events = {}
    FakeVisionAnalyzer.release_events = {}
    FakeVisionAnalyzer.stopped = False

    pipeline.MediaProcessor = FakeMediaProcessor
    pipeline.AudioTranscriber = FakeAudioTranscriber
    pipeline.VisionAnalyzer = FakeVisionAnalyzer
    pipeline.VideoIndexer = FakeVideoIndexer
    pipeline.cleanup_gpu = lambda: None
    try:
        yield
    finally:
        pipeline.MediaProcessor = original_media_processor
        pipeline.AudioTranscriber = original_transcriber
        pipeline.VisionAnalyzer = original_vision_analyzer
        pipeline.VideoIndexer = original_indexer
        pipeline.cleanup_gpu = original_cleanup_gpu


@contextmanager
def patched_attr(target, name, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def make_args(
    root: Path,
    archive_completed_videos: bool = True,
    parallel_requests: int = 2,
    input_dir: Path | None = None,
    completed_dir: Path | None = None,
):
    input_dir = input_dir or (root / "input")
    completed_dir = completed_dir or (root / "completed")
    return argparse.Namespace(
        input=str(input_dir),
        output_dir=str(root / "processed"),
        completed_dir=str(completed_dir),
        archive_completed_videos=archive_completed_videos,
        vlm_model="models/fake.gguf",
        whisper_backend="fake",
        whisper_model="fake",
        whisper_beam_size=1,
        whisper_batch_size=1,
        whisper_batched=False,
        whisper_vad_filter=False,
        whisper_compute_type="float32",
        whisper_device="cpu",
        cuda_dll_dir="",
        max_frames=1,
        batch_size=0,
        parallel_requests=parallel_requests,
        server_parallel=parallel_requests,
        embedding_backend=None,
        embedding_table=None,
        embedding_ctx_size=None,
        embedding_parallel=1,
        verbose=False,
    )


def reset_root(name: str) -> Path:
    root = TEST_ROOT / name
    if root.exists():
        shutil.rmtree(root)
    (root / "input").mkdir(parents=True, exist_ok=True)
    return root


def make_video(root: Path, name: str, content: bytes = b"fake video") -> Path:
    path = root / "input" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_path(root: Path, video_path: Path) -> Path:
    return Path(pipeline.analysis_result_path(str(root / "processed"), str(video_path)))


def run_batch(root: Path, videos, args=None):
    args = args or make_args(root)
    pipeline.process_batch([str(video) for video in videos], args)
    return args


def run_cli(root: Path, extra_args=None, include_completed_dir: bool = True, input_dir: Path | None = None):
    input_dir = input_dir or (root / "input")
    argv = [
        "main.py",
        "--input",
        str(input_dir),
        "--output_dir",
        str(root / "processed"),
        "--vlm_model",
        "models/fake.gguf",
    ]
    if include_completed_dir:
        argv.extend(["--completed_dir", str(root / "completed")])
    if extra_args:
        argv.extend(extra_args)

    original_argv = sys.argv
    sys.argv = argv
    try:
        pipeline.main()
    finally:
        sys.argv = original_argv


def test_success_archives_deletes_source_and_indexes():
    root = reset_root("success")
    video = make_video(root, "ok.mp4")

    with patched_pipeline():
        args = run_batch(root, [video])

    result = result_path(root, video)
    archived = root / "completed" / "ok.mp4"
    data = read_json(result)

    assert not video.exists()
    assert archived.exists()
    assert data["video_path"] == str(archived)
    assert pipeline.is_valid_analysis_result(str(result))
    assert FakeVideoIndexer.indexed_paths == [str(result)]
    assert args.archive_completed_videos is True


def test_vlm_exception_leaves_source_without_json_or_index():
    root = reset_root("vlm_exception")
    video = make_video(root, "bad.mp4")
    FakeVisionAnalyzer.behaviors = {"bad": "failure"}

    with patched_pipeline():
        FakeVisionAnalyzer.behaviors = {"bad": "failure"}
        run_batch(root, [video])

    assert video.exists()
    assert not result_path(root, video).exists()
    assert FakeVideoIndexer.indexed_paths == []


def test_failed_vlm_result_leaves_source_without_json():
    root = reset_root("failed_result")
    video = make_video(root, "failed-result.mp4")

    with patched_pipeline():
        FakeVisionAnalyzer.behaviors = {"failed-result": "failed_result"}
        run_batch(root, [video])

    assert video.exists()
    assert not result_path(root, video).exists()
    assert FakeVideoIndexer.indexed_paths == []


def test_no_archive_flag_keeps_source_and_json_points_to_input():
    root = reset_root("no_archive")
    video = make_video(root, "keep.mp4")
    args = make_args(root, archive_completed_videos=False)

    with patched_pipeline():
        run_batch(root, [video], args)

    result = result_path(root, video)
    data = read_json(result)

    assert video.exists()
    assert not (root / "completed" / "keep.mp4").exists()
    assert data["video_path"] == str(video)
    assert FakeVideoIndexer.indexed_paths == [str(result)]


def test_default_completed_dir_uses_indexed_videos_inside_input():
    root = reset_root("default_completed_dir")
    source_dir = root / "source videos"
    video = make_video(root, "outside.mp4")
    source_dir.mkdir(parents=True, exist_ok=True)
    relocated_video = source_dir / video.name
    shutil.move(str(video), relocated_video)

    args = make_args(
        root,
        input_dir=source_dir,
        completed_dir=Path("data") / "completed_videos",
    )

    with patched_pipeline():
        run_batch(root, [relocated_video], args)

    result = result_path(root, relocated_video)
    archived = source_dir / "indexed-videos" / "outside.mp4"
    data = read_json(result)

    assert not relocated_video.exists()
    assert archived.exists()
    assert data["video_path"] == str(archived)


def test_explicit_completed_dir_override_is_still_honored():
    root = reset_root("explicit_completed_override")
    source_dir = root / "external-input"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    video = nested_dir / "keep-override.mp4"
    video.write_bytes(b"fake video")

    args = make_args(
        root,
        input_dir=source_dir,
        completed_dir=root / "manual-completed",
    )

    with patched_pipeline():
        run_batch(root, [video], args)

    result = result_path(root, video)
    archived = root / "manual-completed" / "nested" / "keep-override.mp4"
    data = read_json(result)

    assert not video.exists()
    assert archived.exists()
    assert data["video_path"] == str(archived)


def test_cli_default_archive_dir_is_skipped_during_discovery():
    root = reset_root("skip_default_archive_dir")
    source_dir = root / "source videos"
    archive_dir = source_dir / "indexed-videos"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_video = archive_dir / "done.mp4"
    archived_video.write_bytes(b"archived video")
    result = result_path(root, archived_video)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "video_path": str(archived_video),
                "summary": "valid existing summary",
                "segments": [{"start_frame": 0, "end_frame": 0, "start_time": 0.0, "end_time": 1.0, "description": "valid"}],
                "analysis": [],
                "timings": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with patched_pipeline():
        run_cli(root, include_completed_dir=False, input_dir=source_dir)

    assert archived_video.exists()
    assert not (archive_dir / "indexed-videos" / "done.mp4").exists()


def test_existing_valid_json_is_archived_and_source_removed():
    root = reset_root("existing_valid")
    video = make_video(root, "already.mp4")
    result = result_path(root, video)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "video_path": str(video),
                "summary": "valid existing summary",
                "segments": [{"start_frame": 0, "end_frame": 0, "start_time": 0.0, "end_time": 1.0, "description": "valid"}],
                "analysis": [],
                "timings": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    args = make_args(root)
    finalized = pipeline.finalize_existing_analysis(str(video), args)
    data = read_json(Path(finalized))
    archived = root / "completed" / "already.mp4"

    assert finalized == str(result)
    assert not video.exists()
    assert archived.exists()
    assert data["video_path"] == str(archived)
    assert data["timings"]["source_video_path"] == str(video)


def test_existing_invalid_json_is_not_considered_complete():
    root = reset_root("existing_invalid")
    video = make_video(root, "invalid.mp4")
    result = result_path(root, video)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps({"video_path": str(video), "summary": "Structured analysis failed.", "segments": []}),
        encoding="utf-8",
    )

    assert not pipeline.is_valid_analysis_result(str(result))


def test_existing_invalid_json_is_reprocessed_by_main():
    root = reset_root("existing_invalid_reprocess")
    video = make_video(root, "invalid-main.mp4")
    result = result_path(root, video)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps({"video_path": str(video), "summary": "Structured analysis failed.", "segments": []}),
        encoding="utf-8",
    )

    with patched_pipeline():
        run_cli(root)

    archived = root / "completed" / "invalid-main.mp4"
    data = read_json(result)
    assert not video.exists()
    assert archived.exists()
    assert data["video_path"] == str(archived)
    assert "valid summary" in data["summary"]
    assert FakeVideoIndexer.indexed_paths == [str(result)]


def test_completed_destination_collision_uses_suffix():
    root = reset_root("collision")
    video = make_video(root, "same.mp4", b"new")
    existing = root / "completed" / "same.mp4"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"old")

    meta = VideoMetadata(
        video_path=str(video),
        base_dir=str(root / "processed" / "same"),
        duration=1.0,
        transcription="fake",
        summary="valid summary",
        frames=[FrameMetadata(path=str(root / "processed" / "same" / "frames" / "frame_0001.jpg"), timestamp=0.0, index=0)],
        segments=[SegmentMetadata(0, 0, 0.0, 1.0, "valid segment")],
    )
    Path(meta.frames[0].path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta.frames[0].path).write_bytes(b"frame")

    args = make_args(root)
    result = pipeline.finalize_successful_video(meta, args)
    suffixed = root / "completed" / "same_1.mp4"

    assert not video.exists()
    assert existing.read_bytes() == b"old"
    assert suffixed.read_bytes() == b"new"
    assert read_json(Path(result))["video_path"] == str(suffixed)


def test_archive_failure_keeps_source_without_json_or_index():
    root = reset_root("archive_failure")
    video = make_video(root, "copy-fails.mp4")

    def failing_copy(original_path, args):
        raise OSError("simulated copy failure")

    with patched_pipeline(), patched_attr(pipeline, "copy_video_to_completed", failing_copy):
        run_batch(root, [video])

    assert video.exists()
    assert not result_path(root, video).exists()
    assert FakeVideoIndexer.indexed_paths == []


def test_write_failure_after_copy_keeps_source_and_does_not_index():
    root = reset_root("write_failure_after_copy")
    video = make_video(root, "write-fails.mp4")

    def failing_write(meta):
        raise OSError("simulated JSON write failure")

    with patched_pipeline(), patched_attr(pipeline, "write_analysis_result", failing_write):
        run_batch(root, [video])

    copied = root / "completed" / "write-fails.mp4"
    assert video.exists()
    assert copied.exists()
    assert not result_path(root, video).exists()
    assert FakeVideoIndexer.indexed_paths == []


def test_remove_failure_after_json_keeps_source_and_does_not_index_until_next_run():
    root = reset_root("remove_failure_after_json")
    video = make_video(root, "remove-fails.mp4")

    def failing_remove(original_path, archived_path, args):
        raise OSError("simulated source delete failure")

    with patched_pipeline(), patched_attr(pipeline, "remove_original_video", failing_remove):
        run_batch(root, [video])

    result = result_path(root, video)
    archived = root / "completed" / "remove-fails.mp4"
    data = read_json(result)
    assert video.exists()
    assert archived.exists()
    assert data["video_path"] == str(archived)
    assert pipeline.is_valid_analysis_result(str(result))
    assert FakeVideoIndexer.indexed_paths == []


def test_main_recovers_valid_json_with_source_left_after_delete_failure():
    root = reset_root("recover_delete_failure")
    video = make_video(root, "recover.mp4")
    result = result_path(root, video)
    archived = root / "completed" / "recover.mp4"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(b"already copied")
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "video_path": str(archived),
                "summary": "valid existing summary",
                "segments": [{"start_frame": 0, "end_frame": 0, "start_time": 0.0, "end_time": 1.0, "description": "valid"}],
                "analysis": [],
                "timings": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with patched_pipeline():
        run_cli(root)

    data = read_json(result)
    assert not video.exists()
    assert archived.exists()
    assert data["video_path"] == str(root / "completed" / "recover_1.mp4")
    assert Path(data["video_path"]).exists()
    assert FakeVideoIndexer.indexed_paths == [str(result)]


def test_async_finalizer_does_not_block_next_vlm_submit():
    root = reset_root("async_finalizer")
    first = make_video(root, "first.mp4")
    second = make_video(root, "second.mp4")
    args = make_args(root, parallel_requests=1)
    original_finalize = pipeline.finalize_successful_video
    finalizer_started = threading.Event()
    allow_finalizer = threading.Event()
    second_started = threading.Event()

    def slow_finalize(meta, args):
        source_name = Path(meta.video_path).stem
        if source_name == "first":
            finalizer_started.set()
            assert allow_finalizer.wait(timeout=10)
        return original_finalize(meta, args)

    with patched_pipeline(), patched_attr(pipeline, "finalize_successful_video", slow_finalize):
        FakeVisionAnalyzer.started_events = {"second": second_started}
        def release_when_second_starts():
            assert finalizer_started.wait(timeout=10)
            assert second_started.wait(timeout=10)
            allow_finalizer.set()

        releaser = threading.Thread(target=release_when_second_starts)
        releaser.start()
        run_batch(root, [first, second], args)
        releaser.join(timeout=10)

    assert finalizer_started.is_set()
    assert second_started.is_set()
    assert not first.exists()
    assert not second.exists()
    assert result_path(root, first).exists()
    assert result_path(root, second).exists()


def test_worker_keyboard_interrupt_saves_completed_and_leaves_active_for_retry():
    root = reset_root("worker_interrupt")
    done = make_video(root, "done.mp4")
    interrupted = make_video(root, "interrupt.mp4")
    args = make_args(root, parallel_requests=2)

    with patched_pipeline():
        FakeVisionAnalyzer.delays = {"done": 0.0, "interrupt": 0.2}
        FakeVisionAnalyzer.behaviors = {"interrupt": "keyboard_interrupt"}
        run_batch(root, [done, interrupted], args)

    assert not done.exists()
    assert (root / "completed" / "done.mp4").exists()
    assert result_path(root, done).exists()
    assert interrupted.exists()
    assert not result_path(root, interrupted).exists()
    assert FakeVideoIndexer.indexed_paths == []
    assert FakeVisionAnalyzer.stopped is True


def test_main_thread_keyboard_interrupt_during_wait_keeps_active_for_retry():
    root = reset_root("main_thread_interrupt")
    active = make_video(root, "active.mp4")
    args = make_args(root, parallel_requests=1)
    active_started = threading.Event()
    release_active = threading.Event()
    original_wait = pipeline.concurrent.futures.wait
    state = {"raised": False}

    def interrupting_wait(fs, *wait_args, **wait_kwargs):
        if not state["raised"] and wait_kwargs.get("return_when") == pipeline.concurrent.futures.FIRST_COMPLETED:
            assert active_started.wait(timeout=10)
            state["raised"] = True
            raise KeyboardInterrupt()
        return original_wait(fs, *wait_args, **wait_kwargs)

    with patched_pipeline(), patched_attr(pipeline.concurrent.futures, "wait", interrupting_wait):
        FakeVisionAnalyzer.started_events = {"active": active_started}
        FakeVisionAnalyzer.release_events = {"active": release_active}
        run_batch(root, [active], args)
        release_active.set()

    assert state["raised"] is True
    assert active.exists()
    assert not result_path(root, active).exists()
    assert FakeVideoIndexer.indexed_paths == []
    assert FakeVisionAnalyzer.stopped is True


def test_stage1_frame_failure_aborts_batch_but_keeps_source_without_json():
    root = reset_root("stage1_failure")
    video = make_video(root, "framefail.mp4")

    with patched_pipeline():
        FakeMediaProcessor.fail_frames = {"framefail"}
        try:
            run_batch(root, [video])
        except RuntimeError as exc:
            assert "frame extraction" in str(exc)
        else:
            raise AssertionError("Stage 1 frame failure should currently abort the batch")

    assert video.exists()
    assert not result_path(root, video).exists()
    assert FakeVideoIndexer.indexed_paths == []


def test_stage2_transcription_failure_aborts_batch_but_keeps_source_without_json():
    root = reset_root("stage2_failure")
    video = make_video(root, "transcribefail.mp4")

    with patched_pipeline():
        FakeAudioTranscriber.fail_for = {"transcribefail"}
        try:
            run_batch(root, [video])
        except RuntimeError as exc:
            assert "transcription" in str(exc)
        else:
            raise AssertionError("Stage 2 transcription failure should currently abort the batch")

    assert video.exists()
    assert not result_path(root, video).exists()
    assert FakeVideoIndexer.indexed_paths == []


TESTS = [
    test_success_archives_deletes_source_and_indexes,
    test_vlm_exception_leaves_source_without_json_or_index,
    test_failed_vlm_result_leaves_source_without_json,
    test_no_archive_flag_keeps_source_and_json_points_to_input,
    test_default_completed_dir_uses_indexed_videos_inside_input,
    test_explicit_completed_dir_override_is_still_honored,
    test_cli_default_archive_dir_is_skipped_during_discovery,
    test_existing_valid_json_is_archived_and_source_removed,
    test_existing_invalid_json_is_not_considered_complete,
    test_existing_invalid_json_is_reprocessed_by_main,
    test_completed_destination_collision_uses_suffix,
    test_archive_failure_keeps_source_without_json_or_index,
    test_write_failure_after_copy_keeps_source_and_does_not_index,
    test_remove_failure_after_json_keeps_source_and_does_not_index_until_next_run,
    test_main_recovers_valid_json_with_source_left_after_delete_failure,
    test_async_finalizer_does_not_block_next_vlm_submit,
    test_worker_keyboard_interrupt_saves_completed_and_leaves_active_for_retry,
    test_main_thread_keyboard_interrupt_during_wait_keeps_active_for_retry,
    test_stage1_frame_failure_aborts_batch_but_keeps_source_without_json,
    test_stage2_transcription_failure_aborts_batch_but_keeps_source_without_json,
]


def main():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    failures = []
    for test in TESTS:
        name = test.__name__
        try:
            test()
            print(f"[PASS] {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"[FAIL] {name}: {exc}")

    if failures:
        print("\nFailures:")
        for name, exc in failures:
            print(f"- {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    print(f"\nAll {len(TESTS)} pipeline recovery tests passed.")


if __name__ == "__main__":
    main()
