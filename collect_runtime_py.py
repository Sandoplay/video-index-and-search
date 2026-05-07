from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime_py_bundle"

ROOT_RUNTIME_FILES = (
    "main.py",
    "bootstrap_launcher.py",
    "bootstrap_runtime.py",
    "setup_models.py",
    "search.py",
    "web_app.py",
    "run_server.py",
    "reindex_lancedb.py",
    "dedupe_lancedb.py",
    "check_llama_cuda.py",
    "collect_runtime_py.py",
    "bootstrap_config.json",
    "build_bootstrap.ps1",
    "requirements.txt",
    "PROJECT_CONTEXT.md",
    "SESSION_LOGS.md",
    "WEB_EMBEDDING_COMMANDS.md",
)

RUNTIME_DIRS = (
    "src",
    "static",
    "templates",
)

OPTIONAL_RUNTIME_FILE_MAPPINGS = (
    ("dist/bootstrap.exe", "bootstrap.exe"),
)

EXCLUDED_DIR_NAMES = (
    "__pycache__",
    ".pytest_cache",
)

EXCLUDED_SUFFIXES = (
    ".pyc",
    ".pyo",
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


def should_copy_file(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def iter_runtime_files() -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []

    for relative_name in ROOT_RUNTIME_FILES:
        source = REPO_ROOT / relative_name
        if not source.exists():
            raise FileNotFoundError(f"Required runtime file not found: {relative_name}")
        if source.is_dir():
            raise IsADirectoryError(f"Expected a file, got directory: {relative_name}")
        files.append((source, Path(relative_name)))

    for source_name, destination_name in OPTIONAL_RUNTIME_FILE_MAPPINGS:
        source = REPO_ROOT / source_name
        if source.exists():
            if source.is_dir():
                raise IsADirectoryError(f"Expected a file, got directory: {source_name}")
            files.append((source, Path(destination_name)))

    for relative_dir in RUNTIME_DIRS:
        source_dir = REPO_ROOT / relative_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Required runtime directory not found: {relative_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory, got file: {relative_dir}")
        for source in sorted(source_dir.rglob("*")):
            if source.is_file() and should_copy_file(source.relative_to(REPO_ROOT)):
                files.append((source, source.relative_to(REPO_ROOT)))

    return sorted(files, key=lambda item: str(item[1]).lower())


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
    for source, relative_destination in iter_runtime_files():
        destination = output_dir / relative_destination
        copied.append(destination)
        if dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the files needed to run the project into one bundle directory."
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
