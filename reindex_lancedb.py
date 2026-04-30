import argparse
from pathlib import Path
from typing import Iterable, List

from src.indexer import VideoIndexer


def find_analysis_files(data_root: Path) -> List[Path]:
    if data_root.name.startswith("processed"):
        roots = [data_root] if data_root.is_dir() else []
    else:
        roots = [path for path in data_root.glob("processed*") if path.is_dir()]

    files: List[Path] = []
    for root in roots:
        files.extend(root.rglob("analysis_result.json"))
    return sorted(files)


def iter_selected_files(args: argparse.Namespace) -> Iterable[Path]:
    if args.json:
        for value in args.json:
            yield Path(value)
        return

    yield from find_analysis_files(Path(args.data_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex existing analysis_result.json files into LanceDB.")
    parser.add_argument("--data-root", default="data", help="Root containing processed* directories")
    parser.add_argument("--json", action="append", help="Specific analysis_result.json path; can be passed multiple times")
    parser.add_argument("--embedding-backend", choices=["minilm", "qwen3-gguf", "qwen3-4b-gguf"], default=None)
    parser.add_argument("--table", default=None, help="Override LanceDB table name")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Number of texts to embed per request")
    parser.add_argument("--add-batch-size", type=int, default=512, help="Number of records to insert into LanceDB per add call")
    parser.add_argument("--embedding-ctx-size", type=int, default=None, help="llama-server context size for embedding indexing")
    parser.add_argument("--embedding-parallel", type=int, default=2, help="llama-server --parallel slots for embedding indexing")
    args = parser.parse_args()

    json_files = list(iter_selected_files(args))
    if not json_files:
        print("No analysis_result.json files found.")
        return

    print(f"Found {len(json_files)} analysis_result.json file(s).")
    indexer = VideoIndexer(
        embedding_backend=args.embedding_backend,
        table_name=args.table,
        embedding_context_size=args.embedding_ctx_size,
        embedding_parallel=args.embedding_parallel,
    )
    try:
        indexer.index_json_files(
            json_files,
            embedding_batch_size=args.embedding_batch_size,
            add_batch_size=args.add_batch_size,
        )
    finally:
        indexer.close()


if __name__ == "__main__":
    main()
