import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return data


def _video_key(data: Dict[str, Any], json_path: Path) -> str:
    source_metadata = data.get("source_metadata") or {}
    filename = str(source_metadata.get("filename") or "").strip()
    if filename:
        return filename

    video_path = str(data.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).name

    return f"{json_path.parent.name}.mp4"


def _collect_results(root: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(root.rglob("analysis_result.json")):
        data = _load_json(json_path)
        key = _video_key(data, json_path)
        results[key] = data
    return results


def _timings(data: Dict[str, Any]) -> Dict[str, float]:
    raw = data.get("timings") or {}
    return {
        "media_processing_seconds": float(raw.get("media_processing_seconds") or 0.0),
        "transcription_seconds": float(raw.get("transcription_seconds") or 0.0),
        "vlm_task_wall_seconds": float(raw.get("vlm_task_wall_seconds") or 0.0),
        "vlm_server_total_seconds": float(raw.get("vlm_server_total_seconds") or 0.0),
        "vlm_prompt_tokens": float(raw.get("vlm_prompt_tokens") or 0.0),
        "vlm_completion_tokens": float(raw.get("vlm_completion_tokens") or 0.0),
        "stage3_total_wall_seconds": float(raw.get("stage3_total_wall_seconds") or 0.0),
    }


def _segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_segments = data.get("segments") or []
    return [segment for segment in raw_segments if isinstance(segment, dict)]


def _run_overview(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {
            "videos": 0,
            "avg_media_processing_seconds": 0.0,
            "avg_transcription_seconds": 0.0,
            "avg_vlm_task_wall_seconds": 0.0,
            "avg_vlm_server_total_seconds": 0.0,
            "avg_summary_chars": 0.0,
            "avg_segment_count": 0.0,
            "stage3_total_wall_seconds": 0.0,
        }

    timing_rows = [_timings(item) for item in results.values()]
    summaries = [len(str(item.get("summary") or "").strip()) for item in results.values()]
    segment_counts = [len(_segments(item)) for item in results.values()]
    return {
        "videos": float(len(results)),
        "avg_media_processing_seconds": mean(row["media_processing_seconds"] for row in timing_rows),
        "avg_transcription_seconds": mean(row["transcription_seconds"] for row in timing_rows),
        "avg_vlm_task_wall_seconds": mean(row["vlm_task_wall_seconds"] for row in timing_rows),
        "avg_vlm_server_total_seconds": mean(row["vlm_server_total_seconds"] for row in timing_rows),
        "avg_summary_chars": mean(summaries),
        "avg_segment_count": mean(segment_counts),
        "stage3_total_wall_seconds": timing_rows[0]["stage3_total_wall_seconds"],
    }


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _paired_rows(
    left_results: Dict[str, Dict[str, Any]],
    right_results: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    left_keys = set(left_results)
    right_keys = set(right_results)
    shared = sorted(left_keys & right_keys, key=str.lower)
    left_only = sorted(left_keys - right_keys, key=str.lower)
    right_only = sorted(right_keys - left_keys, key=str.lower)
    return shared, left_only, right_only


def build_report(
    left_root: Path,
    right_root: Path,
    left_label: str,
    right_label: str,
) -> str:
    left_results = _collect_results(left_root)
    right_results = _collect_results(right_root)
    shared, left_only, right_only = _paired_rows(left_results, right_results)
    left_overview = _run_overview({key: left_results[key] for key in shared})
    right_overview = _run_overview({key: right_results[key] for key in shared})

    lines: List[str] = [
        "# VLM Run Comparison",
        "",
        f"- Left: `{left_label}` from `{left_root}`",
        f"- Right: `{right_label}` from `{right_root}`",
        f"- Matched videos: `{len(shared)}`",
        f"- Left-only videos: `{len(left_only)}`",
        f"- Right-only videos: `{len(right_only)}`",
        "",
        "## Overview",
        "",
        "| Metric | " + _markdown_escape(left_label) + " | " + _markdown_escape(right_label) + " |",
        "|---|---:|---:|",
        f"| Matched videos | {len(shared)} | {len(shared)} |",
        f"| Stage 3 total wall (s) | {left_overview['stage3_total_wall_seconds']:.2f} | {right_overview['stage3_total_wall_seconds']:.2f} |",
        f"| Avg media processing (s/video) | {left_overview['avg_media_processing_seconds']:.2f} | {right_overview['avg_media_processing_seconds']:.2f} |",
        f"| Avg transcription (s/video) | {left_overview['avg_transcription_seconds']:.2f} | {right_overview['avg_transcription_seconds']:.2f} |",
        f"| Avg VLM task wall (s/video) | {left_overview['avg_vlm_task_wall_seconds']:.2f} | {right_overview['avg_vlm_task_wall_seconds']:.2f} |",
        f"| Avg VLM server active (s/video) | {left_overview['avg_vlm_server_total_seconds']:.2f} | {right_overview['avg_vlm_server_total_seconds']:.2f} |",
        f"| Avg summary chars | {left_overview['avg_summary_chars']:.1f} | {right_overview['avg_summary_chars']:.1f} |",
        f"| Avg segment count | {left_overview['avg_segment_count']:.1f} | {right_overview['avg_segment_count']:.1f} |",
        "",
        "## Per-Video Timing",
        "",
        "| Video | " + _markdown_escape(left_label) + " VLM wall (s) | " + _markdown_escape(right_label) + " VLM wall (s) | "
        + _markdown_escape(left_label) + " prompt tokens | " + _markdown_escape(right_label) + " prompt tokens |",
        "|---|---:|---:|---:|---:|",
    ]

    for key in shared:
        left_timing = _timings(left_results[key])
        right_timing = _timings(right_results[key])
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(key),
                    f"{left_timing['vlm_task_wall_seconds']:.2f}",
                    f"{right_timing['vlm_task_wall_seconds']:.2f}",
                    str(int(left_timing["vlm_prompt_tokens"])),
                    str(int(right_timing["vlm_prompt_tokens"])),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Side-By-Side Content", ""])

    for key in shared:
        left_data = left_results[key]
        right_data = right_results[key]
        left_segments = _segments(left_data)
        right_segments = _segments(right_data)
        lines.extend(
            [
                f"### {key}",
                "",
                f"**{left_label} summary**",
                "",
                _safe_text(left_data.get("summary")) or "_empty_",
                "",
                f"**{right_label} summary**",
                "",
                _safe_text(right_data.get("summary")) or "_empty_",
                "",
                f"**{left_label} segments**",
                "",
            ]
        )
        if left_segments:
            for index, segment in enumerate(left_segments, 1):
                lines.append(
                    f"{index}. `{float(segment.get('start_time') or 0.0):.3f}s - {float(segment.get('end_time') or 0.0):.3f}s` "
                    f"{_safe_text(segment.get('description'))}"
                )
        else:
            lines.append("_none_")

        lines.extend(["", f"**{right_label} segments**", ""])
        if right_segments:
            for index, segment in enumerate(right_segments, 1):
                lines.append(
                    f"{index}. `{float(segment.get('start_time') or 0.0):.3f}s - {float(segment.get('end_time') or 0.0):.3f}s` "
                    f"{_safe_text(segment.get('description'))}"
                )
        else:
            lines.append("_none_")
        lines.append("")

    if left_only:
        lines.extend(["## Left-Only Videos", ""])
        lines.extend(f"- {name}" for name in left_only)
        lines.append("")

    if right_only:
        lines.extend(["## Right-Only Videos", ""])
        lines.extend(f"- {name}" for name in right_only)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two VLM analysis_result.json runs side-by-side.")
    parser.add_argument("--left-root", required=True, help="Directory containing analysis_result.json files for the first run")
    parser.add_argument("--right-root", required=True, help="Directory containing analysis_result.json files for the second run")
    parser.add_argument("--left-label", default="Run A", help="Display label for the first run")
    parser.add_argument("--right-label", default="Run B", help="Display label for the second run")
    parser.add_argument("--output", default="data/vlm_compare_report.md", help="Output markdown report path")
    args = parser.parse_args()

    left_root = Path(args.left_root)
    right_root = Path(args.right_root)
    output_path = Path(args.output)

    if not left_root.exists():
        raise FileNotFoundError(f"Left root does not exist: {left_root}")
    if not right_root.exists():
        raise FileNotFoundError(f"Right root does not exist: {right_root}")

    report = build_report(
        left_root=left_root,
        right_root=right_root,
        left_label=args.left_label,
        right_label=args.right_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
