import shutil
from pathlib import Path

import compare_vlm_runs


TEST_ROOT = Path("data/temp_compare_vlm_runs_tests")


def reset_root() -> Path:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_ROOT


def write_result(root: Path, folder: str, video_name: str, summary: str, segment_description: str, vlm_wall: float) -> None:
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("analysis_result.json").write_text(
        (
            "{\n"
            f'  "video_path": ".\\\\videos\\\\{video_name}",\n'
            f'  "source_metadata": {{"filename": "{video_name}", "filesize_bytes": 123, "duration_seconds": 1.0}},\n'
            f'  "summary": "{summary}",\n'
            '  "segments": [\n'
            "    {\n"
            '      "start_frame": 0,\n'
            '      "end_frame": 0,\n'
            '      "start_time": 0.0,\n'
            '      "end_time": 1.0,\n'
            f'      "description": "{segment_description}"\n'
            "    }\n"
            "  ],\n"
            f'  "timings": {{"vlm_task_wall_seconds": {vlm_wall}, "vlm_server_total_seconds": {vlm_wall - 1.0}, "stage3_total_wall_seconds": 10.0, "vlm_prompt_tokens": 100}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )


def test_build_report_matches_videos_and_renders_side_by_side_sections() -> None:
    root = reset_root()
    left = root / "left"
    right = root / "right"

    write_result(left, "clip", "clip.mp4", "left summary", "left segment", 12.5)
    write_result(right, "clip", "clip.mp4", "right summary", "right segment", 8.0)

    report = compare_vlm_runs.build_report(left, right, "9B", "4B")

    assert "# VLM Run Comparison" in report
    assert "| Stage 3 total wall (s) | 10.00 | 10.00 |" in report
    assert "### clip.mp4" in report
    assert "**9B summary**" in report
    assert "left summary" in report
    assert "**4B summary**" in report
    assert "right summary" in report
    assert "left segment" in report
    assert "right segment" in report


def main() -> None:
    tests = [
        test_build_report_matches_videos_and_renders_side_by_side_sections,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"[FAIL] {test.__name__}: {exc}")

    if failures:
        raise SystemExit(1)

    print(f"\nAll {len(tests)} compare VLM run tests passed.")


if __name__ == "__main__":
    main()
