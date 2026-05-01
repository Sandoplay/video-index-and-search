import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.search_service import SearchService


DEFAULT_CONFIGS = [
    {
        "label": "MiniLM",
        "backend": "minilm",
        "table": "video_metadata_74_minilm",
    },
    {
        "label": "Qwen3 0.6B",
        "backend": "qwen3-gguf",
        "table": "video_metadata_74_qwen3_1024",
    },
    {
        "label": "Qwen3 4B",
        "backend": "qwen3-4b-gguf",
        "table": "video_metadata_74_qwen3_4b_2560",
    },
]


def _matches_expected(video_path: str, expected_video: str) -> bool:
    return Path(video_path).name.lower() == expected_video.lower()


def _rank_for_expected(results: List[Dict[str, Any]], expected_video: str) -> Optional[int]:
    for index, result in enumerate(results, 1):
        if _matches_expected(str(result.get("video_path") or ""), expected_video):
            return index
    return None


def _load_eval_cases(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return _normalize_flat_cases(data)
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return _normalize_template_cases(data["videos"])

    raise ValueError("Eval file must contain either a flat JSON list or an object with a 'videos' list.")


def _normalize_flat_cases(raw_cases: List[Any]) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    for index, raw_case in enumerate(raw_cases, 1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case #{index} must be a JSON object.")
        query = str(raw_case.get("query") or "").strip()
        expected_video = str(raw_case.get("expected_video") or "").strip()
        if not query or not expected_video:
            raise ValueError(f"Case #{index} must include query and expected_video.")
        cases.append({"query": query, "expected_video": expected_video})
    return cases


def _normalize_template_cases(videos: List[Any]) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    for video_index, raw_video in enumerate(videos, 1):
        if not isinstance(raw_video, dict):
            raise ValueError(f"Video entry #{video_index} must be a JSON object.")

        expected_video = str(raw_video.get("expected_video") or "").strip()
        raw_queries = raw_video.get("queries")
        if not expected_video:
            raise ValueError(f"Video entry #{video_index} must include expected_video.")
        if not isinstance(raw_queries, list):
            raise ValueError(f"Video entry #{video_index} must include a queries list.")

        for query_index, raw_query in enumerate(raw_queries, 1):
            query = str(raw_query or "").strip()
            if not query:
                continue
            cases.append(
                {
                    "query": query,
                    "expected_video": expected_video,
                    "case_id": f"{video_index}.{query_index}",
                }
            )

    if not cases:
        raise ValueError("No non-empty queries found in the eval template.")
    return cases


def _evaluate_config(
    config: Dict[str, str],
    cases: List[Dict[str, str]],
    item_type: str,
    limit: int,
    lexical_rerank: bool,
) -> Dict[str, Any]:
    print()
    print(f"=== {config['label']} | backend={config['backend']} | table={config['table']} ===")
    service = SearchService(embedding_backend=config["backend"], table_name=config["table"])
    ranks: List[Optional[int]] = []
    details: List[Dict[str, Any]] = []

    try:
        for case in cases:
            results = service.search(
                case["query"],
                item_type=item_type,
                limit=limit,
                lexical_rerank=lexical_rerank,
            )
            rank = _rank_for_expected(results, case["expected_video"])
            top_video = Path(str(results[0].get("video_path") or "")).name if results else "-"
            top_distance = results[0].get("distance") if results else None
            ranks.append(rank)
            details.append(
                {
                    "query": case["query"],
                    "expected_video": case["expected_video"],
                    "rank": rank,
                    "top_video": top_video,
                    "top_distance": top_distance,
                }
            )

            marker = f"rank {rank}" if rank is not None else f"miss@{limit}"
            print(f"{marker:>8} | {case['query']} | expected={case['expected_video']} | top={top_video}")
    finally:
        service.close()

    return {
        "label": config["label"],
        "backend": config["backend"],
        "table": config["table"],
        "summary": _summary_from_ranks(ranks, limit),
        "details": details,
    }


def _summary_from_ranks(ranks: List[Optional[int]], limit: int) -> Dict[str, Any]:
    total = len(ranks)
    hits = [rank for rank in ranks if rank is not None]
    top1 = sum(1 for rank in ranks if rank == 1)
    top3 = sum(1 for rank in ranks if rank is not None and rank <= 3)
    top10 = sum(1 for rank in ranks if rank is not None and rank <= 10)
    top_limit = sum(1 for rank in ranks if rank is not None and rank <= limit)
    mean_rank = sum(hits) / len(hits) if hits else 0.0

    return {
        "cases": total,
        "top1": top1,
        "top3": top3,
        "top10": top10,
        "top_limit": top_limit,
        "mean_rank": mean_rank,
        "misses": total - top_limit,
    }


def _markdown_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _rank_label(rank: Optional[int], limit: int) -> str:
    return str(rank) if rank is not None else f"miss@{limit}"


def _percent(value: int, total: int) -> str:
    return f"{value / total:.1%}" if total else "0.0%"


def _build_markdown_report(
    cases: List[Dict[str, str]],
    results: List[Dict[str, Any]],
    item_type: str,
    limit: int,
    lexical_rerank: bool,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "# Embedding Search Comparison",
        "",
        f"Generated: `{now}`",
        "",
        "## Settings",
        "",
        f"- Cases: `{len(cases)}`",
        f"- Search type: `{item_type}`",
        f"- Limit: `{limit}`",
        f"- Lexical rerank: `{lexical_rerank}`",
        "",
        "## Summary",
        "",
        "| Model | Backend | Table | Top-1 | Top-3 | Top-10 | Hits@limit | Misses | Mean rank for hits |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        summary = result["summary"]
        total = summary["cases"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(result["label"]),
                    _markdown_escape(result["backend"]),
                    _markdown_escape(result["table"]),
                    f"{summary['top1']}/{total} ({_percent(summary['top1'], total)})",
                    f"{summary['top3']}/{total} ({_percent(summary['top3'], total)})",
                    f"{summary['top10']}/{total} ({_percent(summary['top10'], total)})",
                    f"{summary['top_limit']}/{total} ({_percent(summary['top_limit'], total)})",
                    str(summary["misses"]),
                    f"{summary['mean_rank']:.2f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Query Ranks",
            "",
            "| # | Query | Expected video | " + " | ".join(_markdown_escape(result["label"]) for result in results) + " |",
            "|---:|---|---|" + "|".join("---:" for _ in results) + "|",
        ]
    )

    for index, case in enumerate(cases, 1):
        ranks = [
            _rank_label(result["details"][index - 1]["rank"], limit)
            for result in results
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _markdown_escape(case["query"]),
                    _markdown_escape(case["expected_video"]),
                    *ranks,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Top Result Per Query",
            "",
            "| # | Model | Query | Expected video | Rank | Top result | Top distance |",
            "|---:|---|---|---|---:|---|---:|",
        ]
    )

    for result in results:
        for index, detail in enumerate(result["details"], 1):
            top_distance = detail["top_distance"]
            distance_label = f"{float(top_distance):.4f}" if top_distance is not None else "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _markdown_escape(result["label"]),
                        _markdown_escape(detail["query"]),
                        _markdown_escape(detail["expected_video"]),
                        _rank_label(detail["rank"], limit),
                        _markdown_escape(detail["top_video"]),
                        distance_label,
                    ]
                )
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple embedding backends on the same labeled search-evaluation queries."
    )
    parser.add_argument("--cases", default="data/search_eval_sample_10_template.json", help="Eval JSON file")
    parser.add_argument("--type", choices=["summary", "all", "segment", "transcription"], default="all", help="Search type")
    parser.add_argument("--limit", type=int, default=10, help="Search depth to inspect")
    parser.add_argument("--lexical-rerank", action="store_true", help="Enable optional lexical rerank")
    parser.add_argument(
        "--output",
        default="experiments/embedding_search_comparison_2026-05-01.md",
        help="Markdown report output path",
    )
    args = parser.parse_args()

    cases = _load_eval_cases(args.cases)
    print(f"Loaded eval cases: {len(cases)}")
    print(f"Search type: {args.type}; limit: {args.limit}; lexical rerank: {args.lexical_rerank}")

    results = [
        _evaluate_config(
            config=config,
            cases=cases,
            item_type=args.type,
            limit=args.limit,
            lexical_rerank=args.lexical_rerank,
        )
        for config in DEFAULT_CONFIGS
    ]

    report = _build_markdown_report(
        cases=cases,
        results=results,
        item_type=args.type,
        limit=args.limit,
        lexical_rerank=args.lexical_rerank,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print()
    print("=== Summary ===")
    for result in results:
        summary = result["summary"]
        total = summary["cases"]
        print(
            f"{result['label']}: "
            f"Top-1 {summary['top1']}/{total} ({_percent(summary['top1'], total)}), "
            f"Top-3 {summary['top3']}/{total} ({_percent(summary['top3'], total)}), "
            f"Top-10 {summary['top10']}/{total} ({_percent(summary['top10'], total)}), "
            f"mean rank {summary['mean_rank']:.2f}"
        )
    print(f"Saved markdown report: {output_path}")


if __name__ == "__main__":
    main()
