import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.search_service import SearchService


def _matches_expected(video_path: str, expected_video: str) -> bool:
    return Path(video_path).name.lower() == expected_video.lower()


def _rank_for_expected(results: List[Dict[str, Any]], expected_video: str) -> Optional[int]:
    for index, result in enumerate(results, 1):
        if _matches_expected(str(result.get("video_path") or ""), expected_video):
            return index
    return None


def _load_cases(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cases = json.load(handle)

    if not isinstance(cases, list):
        raise ValueError("Eval file must contain a JSON list.")

    normalized: List[Dict[str, str]] = []
    for index, case in enumerate(cases, 1):
        query = str(case.get("query") or "").strip()
        expected_video = str(case.get("expected_video") or "").strip()
        if not query or not expected_video:
            raise ValueError(f"Eval case #{index} must include query and expected_video.")
        normalized.append({"query": query, "expected_video": expected_video})
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic search against a small labeled query set.")
    parser.add_argument("--cases", default="data/search_eval_queries.json", help="JSON file with query/expected_video cases")
    parser.add_argument("--embedding-backend", choices=["minilm", "qwen3-gguf", "qwen3-4b-gguf"], default=None)
    parser.add_argument("--table", default=None, help="Override LanceDB table name")
    parser.add_argument("--type", choices=["summary", "all", "segment", "transcription"], default="all", help="Search type")
    parser.add_argument("--limit", type=int, default=10, help="Search depth to inspect")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    service = SearchService(embedding_backend=args.embedding_backend, table_name=args.table)
    ranks: List[Optional[int]] = []

    try:
        for case in cases:
            results = service.search(case["query"], item_type=args.type, limit=args.limit)
            rank = _rank_for_expected(results, case["expected_video"])
            ranks.append(rank)

            marker = f"rank {rank}" if rank is not None else f"miss@{args.limit}"
            top_video = Path(str(results[0].get("video_path") or "")).name if results else "-"
            print(f"{marker:>8} | {case['query']} | expected={case['expected_video']} | top={top_video}")
    finally:
        service.close()

    total = len(ranks)
    top1 = sum(1 for rank in ranks if rank == 1)
    top3 = sum(1 for rank in ranks if rank is not None and rank <= 3)
    top10 = sum(1 for rank in ranks if rank is not None and rank <= 10)
    mean_rank_values = [rank for rank in ranks if rank is not None]
    mean_rank = sum(mean_rank_values) / len(mean_rank_values) if mean_rank_values else 0.0

    print()
    print(f"Cases: {total}")
    print(f"Top-1:  {top1}/{total} ({top1 / total:.1%})")
    print(f"Top-3:  {top3}/{total} ({top3 / total:.1%})")
    print(f"Top-10: {top10}/{total} ({top10 / total:.1%})")
    print(f"Mean rank for hits: {mean_rank:.2f}")


if __name__ == "__main__":
    main()
