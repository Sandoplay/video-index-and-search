from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
BIN_DIR = REPO_ROOT / "bin"
LLAMA_CPP_DIR = BIN_DIR / "llama-cpp"
LLAMA_SERVER_PATH = LLAMA_CPP_DIR / "llama-server.exe"
LLAMA_CPP_RELEASE_TAG = "b8064"
LLAMA_CPP_LATEST_RELEASE_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_CPP_DOWNLOAD_DIR = BIN_DIR / "downloads"
FFMPEG_DOWNLOAD_DIR = BIN_DIR / "downloads"


@dataclass(frozen=True)
class FfmpegArchive:
    url: str
    filename: str
    display_size: str
    size_gb: float
    purpose: str
    required_files: tuple[Path, ...]


@dataclass(frozen=True)
class LlamaCppArchive:
    url: str
    filename: str
    display_size: str
    size_gb: float
    purpose: str
    required_files: tuple[Path, ...]


@dataclass(frozen=True)
class ModelFile:
    repo_id: str
    filename: str
    target_path: Path
    display_size: str
    size_gb: float
    purpose: str


FFMPEG_ARCHIVE = FfmpegArchive(
    url=(
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-win64-gpl.zip"
    ),
    filename="ffmpeg-master-latest-win64-gpl.zip",
    display_size="~170 MB",
    size_gb=0.17,
    purpose="FFmpeg static Windows build from GitHub",
    required_files=(BIN_DIR / "ffmpeg.exe", BIN_DIR / "ffprobe.exe"),
)


def llama_cpp_archives_for_release(release_tag: str) -> tuple[LlamaCppArchive, ...]:
    return (
        LlamaCppArchive(
            url=(
                "https://github.com/ggml-org/llama.cpp/releases/download/"
                f"{release_tag}/llama-{release_tag}-bin-win-cuda-13.1-x64.zip"
            ),
            filename=f"llama-{release_tag}-bin-win-cuda-13.1-x64.zip",
            display_size="~170 MB",
            size_gb=0.17,
            purpose="llama.cpp Windows CUDA 13.1 binaries",
            required_files=(LLAMA_SERVER_PATH, LLAMA_CPP_DIR / "ggml-cuda.dll"),
        ),
        LlamaCppArchive(
            url=(
                "https://github.com/ggml-org/llama.cpp/releases/download/"
                f"{release_tag}/cudart-llama-bin-win-cuda-13.1-x64.zip"
            ),
            filename=f"cudart-llama-bin-win-cuda-13.1-x64.zip",
            display_size="~384 MB",
            size_gb=0.384,
            purpose="CUDA 13.1 runtime DLLs for llama.cpp",
            required_files=(
                LLAMA_CPP_DIR / "cublas64_13.dll",
                LLAMA_CPP_DIR / "cublasLt64_13.dll",
                LLAMA_CPP_DIR / "cudart64_13.dll",
            ),
        ),
    )


LLAMA_CPP_ARCHIVES = llama_cpp_archives_for_release(LLAMA_CPP_RELEASE_TAG)


DEFAULT_MODELS = (
    ModelFile(
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        target_path=MODELS_DIR / "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        display_size="5.97 GB",
        size_gb=5.97,
        purpose="VLM",
    ),
    ModelFile(
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        filename="mmproj-F16.gguf",
        target_path=MODELS_DIR / "mmproj-F16.gguf",
        display_size="918 MB",
        size_gb=0.918,
        purpose="VLM projector",
    ),
    ModelFile(
        repo_id="Qwen/Qwen3-Embedding-4B-GGUF",
        filename="Qwen3-Embedding-4B-Q8_0.gguf",
        target_path=MODELS_DIR / "Qwen3-Embedding-4B-Q8_0.gguf",
        display_size="4.28 GB",
        size_gb=4.28,
        purpose="Qwen3 embedding",
    ),
)


def has_huggingface_hub() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def missing_ffmpeg_archive(archive: FfmpegArchive = FFMPEG_ARCHIVE) -> FfmpegArchive | None:
    return archive if any(not path.exists() for path in archive.required_files) else None


def print_ffmpeg_status(archive: FfmpegArchive = FFMPEG_ARCHIVE) -> None:
    missing = [path for path in archive.required_files if not path.exists()]
    status = "present" if not missing else "missing"
    print("FFmpeg files:")
    print(f"  [{status}] {archive.purpose}")
    for path in missing:
        print(f"    - {path.relative_to(REPO_ROOT)}")


def missing_llama_cpp_archives(
    archives: Iterable[LlamaCppArchive] = LLAMA_CPP_ARCHIVES,
) -> list[LlamaCppArchive]:
    return [archive for archive in archives if any(not path.exists() for path in archive.required_files)]


def print_llama_cpp_status(archives: Iterable[LlamaCppArchive] = LLAMA_CPP_ARCHIVES) -> None:
    print("llama.cpp files:")
    for archive in archives:
        missing = [path for path in archive.required_files if not path.exists()]
        status = "present" if not missing else "missing"
        print(f"  [{status}] {archive.purpose}")
        for path in missing:
            print(f"    - {path.relative_to(REPO_ROOT)}")


def missing_models(models: Iterable[ModelFile] = DEFAULT_MODELS) -> list[ModelFile]:
    return [model for model in models if not model.target_path.exists()]


def print_model_status(models: Iterable[ModelFile] = DEFAULT_MODELS) -> None:
    print("Default model files:")
    for model in models:
        status = "present" if model.target_path.exists() else "missing"
        rel_path = model.target_path.relative_to(REPO_ROOT)
        print(f"  [{status}] {model.purpose}: {rel_path} ({model.display_size})")


def prompt_download(total_size_gb: float) -> bool:
    answer = input(f"\nDownload missing default models now? (~{total_size_gb:.1f} GB) [Y/n]: ").strip().lower()
    return answer in {"", "y", "yes"}


def prompt_download_llama_cpp(total_size_gb: float) -> bool:
    answer = input(f"\nDownload missing llama.cpp files now? (~{total_size_gb:.1f} GB) [Y/n]: ").strip().lower()
    return answer in {"", "y", "yes"}


def prompt_download_ffmpeg(total_size_gb: float) -> bool:
    answer = input(f"\nDownload missing FFmpeg files now? (~{total_size_gb:.1f} GB) [Y/n]: ").strip().lower()
    return answer in {"", "y", "yes"}


def prompt_llama_cpp_release() -> str:
    print("\nSelect llama.cpp version:")
    print(f"  [1] Default pinned release ({LLAMA_CPP_RELEASE_TAG})")
    print("  [2] Latest GitHub release")
    answer = input("Choose [1/2] (default 1): ").strip().lower()
    return "latest" if answer in {"2", "latest", "l"} else LLAMA_CPP_RELEASE_TAG


def fetch_latest_llama_cpp_release_tag() -> str:
    request = urllib.request.Request(
        LLAMA_CPP_LATEST_RELEASE_API_URL,
        headers={"User-Agent": "ml-vid-indexer-search-setup/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    tag_name = data.get("tag_name")
    if not isinstance(tag_name, str) or not tag_name:
        raise RuntimeError("Could not determine latest llama.cpp release tag from GitHub.")
    return tag_name


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ml-vid-indexer-search-setup/1.0"})
    with urllib.request.urlopen(request) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        chunk_size = 1024 * 1024
        with destination.open("wb") as file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    percent = downloaded / total_bytes * 100
                    print(f"\r  {downloaded / (1024 * 1024):.1f}/{total_bytes / (1024 * 1024):.1f} MB ({percent:.1f}%)", end="")
                else:
                    print(f"\r  {downloaded / (1024 * 1024):.1f} MB", end="")
    print()
    return destination


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.infolist() if member.filename and not member.is_dir()]
        path_parts = [Path(member.filename).parts for member in members]
        first_parts = {parts[0] for parts in path_parts if len(parts) > 1}
        has_root_files = any(len(parts) == 1 for parts in path_parts)
        strip_prefix = next(iter(first_parts)) if len(first_parts) == 1 and not has_root_files else None

        for member in members:
            member_path = Path(member.filename)
            relative_parts = member_path.parts
            if strip_prefix and relative_parts and relative_parts[0] == strip_prefix:
                relative_parts = relative_parts[1:]
            if not relative_parts:
                continue
            target = destination.joinpath(*relative_parts).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe zip path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                output.write(source.read())


def extract_binaries_from_zip(zip_path: Path, destination: Path, binary_names: Iterable[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {name.lower(): name for name in binary_names}
    extracted: set[str] = set()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename:
                continue
            member_path = Path(member.filename)
            if len(member_path.parts) < 2 or member_path.parts[-2].lower() != "bin":
                continue

            binary_name = wanted.get(member_path.name.lower())
            if not binary_name:
                continue

            target = destination / binary_name
            with archive.open(member) as source, target.open("wb") as output:
                output.write(source.read())
            extracted.add(binary_name.lower())

    missing = sorted(set(wanted) - extracted)
    if missing:
        raise RuntimeError(f"Archive did not contain expected FFmpeg binaries: {', '.join(missing)}")


def download_ffmpeg_archive(archive: FfmpegArchive = FFMPEG_ARCHIVE) -> None:
    zip_path = FFMPEG_DOWNLOAD_DIR / archive.filename
    print(f"\nDownloading {archive.purpose} ({archive.display_size})...")
    download_file(archive.url, zip_path)
    print(f"Extracting FFmpeg binaries to: {BIN_DIR.relative_to(REPO_ROOT)}")
    extract_binaries_from_zip(zip_path, BIN_DIR, ("ffmpeg.exe", "ffprobe.exe"))
    missing = [path for path in archive.required_files if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        raise RuntimeError(f"Archive extracted, but expected file(s) are still missing: {missing_list}")


def download_llama_cpp_archive(archive: LlamaCppArchive) -> None:
    zip_path = LLAMA_CPP_DOWNLOAD_DIR / archive.filename
    print(f"\nDownloading {archive.purpose} ({archive.display_size})...")
    download_file(archive.url, zip_path)
    print(f"Extracting to: {LLAMA_CPP_DIR.relative_to(REPO_ROOT)}")
    safe_extract_zip(zip_path, LLAMA_CPP_DIR)
    missing = [path for path in archive.required_files if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        raise RuntimeError(f"Archive extracted, but expected file(s) are still missing: {missing_list}")


def download_model(model: ModelFile) -> Path:
    print(f"\nDownloading {model.filename} from {model.repo_id}...")
    if not has_huggingface_hub():
        encoded_filename = urllib.parse.quote(model.filename)
        url = f"https://huggingface.co/{model.repo_id}/resolve/main/{encoded_filename}"
        return download_file(url, model.target_path)

    from huggingface_hub import hf_hub_download

    download_kwargs = {
        "repo_id": model.repo_id,
        "filename": model.filename,
        "local_dir": str(model.target_path.parent),
    }
    if "local_dir_use_symlinks" in inspect.signature(hf_hub_download).parameters:
        download_kwargs["local_dir_use_symlinks"] = False

    path = hf_hub_download(**download_kwargs)
    downloaded_path = Path(path)
    print(f"Downloaded to: {downloaded_path}")
    return downloaded_path


def validate_prerequisites() -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA_CPP_DIR.mkdir(parents=True, exist_ok=True)

    if not has_huggingface_hub():
        print("Notice: huggingface_hub is not installed. Model downloads will use direct Hugging Face URLs.")
        print("You can still install dependencies later with: pip install -r requirements.txt")

    return True


def main() -> int:
    print("Model setup for ml-vid-indexer-search\n")

    prerequisites_ok = validate_prerequisites()
    print_ffmpeg_status()
    print_llama_cpp_status()
    print_model_status()

    missing_ffmpeg = missing_ffmpeg_archive()
    if missing_ffmpeg:
        print(f"\nMissing FFmpeg archive, approximately {missing_ffmpeg.size_gb:.1f} GB total:")
        print(f"  - {missing_ffmpeg.purpose}: {missing_ffmpeg.filename} ({missing_ffmpeg.display_size})")
        if prompt_download_ffmpeg(missing_ffmpeg.size_gb):
            download_ffmpeg_archive(missing_ffmpeg)
        else:
            print("FFmpeg download skipped.")

    missing_archives = missing_llama_cpp_archives()
    if missing_archives:
        selected_release = prompt_llama_cpp_release()
        if selected_release == "latest":
            selected_release = fetch_latest_llama_cpp_release_tag()
            print(f"Latest llama.cpp release: {selected_release}")
        selected_archives = llama_cpp_archives_for_release(selected_release)
        missing_archives = missing_llama_cpp_archives(selected_archives)
        total_size_gb = sum(archive.size_gb for archive in missing_archives)
        print(f"\nMissing llama.cpp archive(s), approximately {total_size_gb:.1f} GB total:")
        for archive in missing_archives:
            print(f"  - {archive.purpose}: {archive.filename} ({archive.display_size})")

        if prompt_download_llama_cpp(total_size_gb):
            for archive in missing_archives:
                download_llama_cpp_archive(archive)
        else:
            print("llama.cpp download skipped.")

    missing = missing_models()
    if not missing:
        if missing_ffmpeg_archive() or missing_llama_cpp_archives():
            print("\nAll default model files are already present.")
        else:
            print("\nAll default FFmpeg, llama.cpp, and model files are already present. Nothing to download.")
        return 0

    total_size_gb = sum(model.size_gb for model in missing)
    print(f"\nMissing {len(missing)} default model file(s), approximately {total_size_gb:.1f} GB total:")
    for model in missing:
        rel_path = model.target_path.relative_to(REPO_ROOT)
        print(f"  - {model.purpose}: {rel_path} ({model.display_size})")

    if not prerequisites_ok:
        return 1

    if not prompt_download(total_size_gb):
        print("Download skipped.")
        return 0

    for model in missing:
        download_model(model)

    print("\nModel setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
