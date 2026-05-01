import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return data


def _video_name(result: Dict[str, Any], analysis_path: Path) -> str:
    raw_video_path = str(result.get("video_path") or "").strip()
    if raw_video_path:
        return Path(raw_video_path).name
    return analysis_path.parent.name


def _is_valid_analysis(result: Dict[str, Any]) -> bool:
    summary = str(result.get("summary") or "").strip()
    segments = result.get("segments")
    has_segments = isinstance(segments, list) and any(isinstance(item, dict) for item in segments)
    return bool(summary or has_segments)


def _segment_template(result: Dict[str, Any], max_segments: int) -> List[Dict[str, Any]]:
    segments = result.get("segments")
    if not isinstance(segments, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for segment in segments[:max_segments]:
        if not isinstance(segment, dict):
            continue
        cleaned.append(
            {
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "description": str(segment.get("description") or "").strip(),
            }
        )
    return cleaned


def _collect_analysis_files(data_root: Path) -> List[Path]:
    return sorted(data_root.rglob("analysis_result.json"))


def _build_sample_item(
    analysis_path: Path,
    result: Dict[str, Any],
    data_root: Path,
    query_slots: int,
    max_segments: int,
) -> Dict[str, Any]:
    video_name = _video_name(result, analysis_path)
    return {
        "expected_video": video_name,
        "analysis_path": str(analysis_path.relative_to(data_root.parent)),
        "summary": str(result.get("summary") or "").strip(),
        "segments": _segment_template(result, max_segments),
        "queries": ["" for _ in range(query_slots)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample analyzed videos and create a query-labeling template for search evaluation."
    )
    parser.add_argument("--data-root", default="data/74-vids", help="Directory with analyzed video folders")
    parser.add_argument("--output", default="data/search_eval_sample_10_template.json", help="Output JSON template")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of videos to sample")
    parser.add_argument("--seed", type=int, default=20260501, help="Random seed for reproducible sampling")
    parser.add_argument("--query-slots", type=int, default=3, help="Blank query slots to create per video")
    parser.add_argument("--max-segments", type=int, default=3, help="Maximum segment descriptions to include")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite it.")
    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1.")
    if args.query_slots < 0:
        raise ValueError("--query-slots must be 0 or greater.")

    valid_items: List[Dict[str, Any]] = []
    skipped = 0

    for analysis_path in _collect_analysis_files(data_root):
        result = _load_json(analysis_path)
        if not _is_valid_analysis(result):
            skipped += 1
            continue
        valid_items.append(
            _build_sample_item(
                analysis_path=analysis_path,
                result=result,
                data_root=data_root,
                query_slots=args.query_slots,
                max_segments=args.max_segments,
            )
        )

    if len(valid_items) < args.sample_size:
        raise ValueError(f"Only found {len(valid_items)} valid analyzed videos, cannot sample {args.sample_size}.")

    rng = random.Random(args.seed)
    sampled_items = rng.sample(valid_items, args.sample_size)
    sampled_items.sort(key=lambda item: item["expected_video"].lower())

    output = {
        "description": "Fill the queries arrays, then use this sample as a labeled search-evaluation set.",
        "data_root": str(data_root),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "valid_analyzed_videos_found": len(valid_items),
        "skipped_invalid_analysis_files": skipped,
        "videos": sampled_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Found valid analyzed videos: {len(valid_items)}")
    print(f"Skipped invalid analysis files: {skipped}")
    print(f"Sample size: {len(sampled_items)}")
    print(f"Seed: {args.seed}")
    print(f"Saved template: {output_path}")
    print()
    print("Selected videos:")
    for item in sampled_items:
        print(f"- {item['expected_video']}")


if __name__ == "__main__":
    main()
