from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime_py_bundle"

RUNTIME_FILES = (
    "main.py",
    "setup_models.py",
    "search.py",
    "web_app.py",
    "run_server.py",
    "reindex_lancedb.py",
    "dedupe_lancedb.py",
    "collect_runtime_py.py",
    "src/embedding_provider.py",
    "src/indexer.py",
    "src/media_processor.py",
    "src/models.py",
    "src/search_service.py",
    "src/transcriber.py",
    "src/vision_analyzer.py",
    "requirements.txt",
)


def resolve_output_dir(raw_output: str | None) -> Path:
    if not raw_output:
        return DEFAULT_OUTPUT_DIR
    output = Path(raw_output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    return output.resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_runtime_py(output_dir: Path, clean: bool = False, dry_run: bool = False) -> list[Path]:
    output_dir = output_dir.resolve()
    repo_root = REPO_ROOT.resolve()

    if output_dir == repo_root:
        raise ValueError("Output directory cannot be the repository root.")

    if clean and output_dir.exists():
        if not is_relative_to(output_dir, repo_root):
            raise ValueError("--clean is only allowed for output directories inside the repository.")
        shutil.rmtree(output_dir)

    copied: list[Path] = []
    for relative_name in RUNTIME_FILES:
        source = REPO_ROOT / relative_name
        if not source.exists():
            raise FileNotFoundError(f"Required Python file not found: {relative_name}")

        destination = output_dir / relative_name
        copied.append(destination)
        if dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy runtime project files into one bundle directory while preserving package paths."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Output directory for the copied Python files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before copying. Only allowed inside the repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be copied without writing anything.",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output)
    copied = collect_runtime_py(output_dir=output_dir, clean=args.clean, dry_run=args.dry_run)

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {len(copied)} runtime files to: {output_dir}")
    for destination in copied:
        print(f"  - {destination.relative_to(output_dir)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
