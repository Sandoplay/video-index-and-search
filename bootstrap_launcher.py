from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_CONFIG = {
    "python_version": "3.12",
    "uv_zip_url": "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip",
    "requirements_path": "requirements.txt",
    "runtime_script": "bootstrap_runtime.py",
}


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = project_root()
BOOTSTRAP_DIR = ROOT / ".bootstrap"
UV_DIR = BOOTSTRAP_DIR / "uv"
UV_EXE = UV_DIR / "uv.exe"
DOWNLOAD_DIR = BOOTSTRAP_DIR / "downloads"
PYTHON_INSTALL_DIR = BOOTSTRAP_DIR / "python"
REQUIREMENTS_STAMP = BOOTSTRAP_DIR / "requirements.sha256"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "bootstrap.log"
CONFIG_PATH = ROOT / "bootstrap_config.json"


class BootstrapRunner:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_PATH.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def run(self) -> int:
        self.log("Preparing local runtime. Progress is written to logs\\bootstrap.log.")
        try:
            config = load_config()
            ensure_uv(config["uv_zip_url"], self.log)
            ensure_python(config["python_version"], self.log)
            ensure_venv(config["python_version"], self.log)
            install_requirements(config["requirements_path"], self.log)
            launch_runtime(config["runtime_script"], self.log)
            self.log("Bootstrap complete. Runtime menu is opening in the browser.")
            return 0
        except Exception as exc:
            self.log(f"Bootstrap failed: {exc}")
            open_log_file()
            return 1
        finally:
            self.log_file.close()


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        config.update(loaded)
    return config


def bootstrap_env() -> dict[str, str]:
    env = os.environ.copy()
    env["UV_PYTHON_INSTALL_DIR"] = str(PYTHON_INSTALL_DIR)
    env["UV_CACHE_DIR"] = str(ROOT / ".uv-cache")
    return env


def run_command(args: list[str], log: callable, env: dict[str, str] | None = None) -> None:
    log("> " + " ".join(args))
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env or bootstrap_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"Command failed with exit code {exit_code}: {' '.join(args)}")


def download_file(url: str, destination: Path, log: callable) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ml-vid-indexer-bootstrap/1.0"})
    with urllib.request.urlopen(request) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        next_report = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        log(f"  {downloaded / 1048576:.1f}/{total / 1048576:.1f} MB")
                    else:
                        log(f"  {downloaded / 1048576:.1f} MB")
                    next_report = downloaded + 25 * 1024 * 1024


def ensure_uv(uv_zip_url: str, log: callable) -> None:
    if UV_EXE.exists():
        log(f"uv already present: {UV_EXE.relative_to(ROOT)}")
        return

    archive_path = DOWNLOAD_DIR / "uv-x86_64-pc-windows-msvc.zip"
    download_file(uv_zip_url, archive_path, log)
    UV_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name.lower()
            if name not in {"uv.exe", "uvx.exe"}:
                continue
            target = UV_DIR / Path(member.filename).name
            with archive.open(member) as source, target.open("wb") as output:
                output.write(source.read())

    if not UV_EXE.exists():
        raise RuntimeError("Downloaded uv archive did not contain uv.exe.")
    log(f"uv extracted to: {UV_EXE.relative_to(ROOT)}")


def ensure_python(python_version: str, log: callable) -> None:
    log(f"Ensuring portable Python {python_version}...")
    run_command(
        [
            str(UV_EXE),
            "python",
            "install",
            python_version,
            "--install-dir",
            str(PYTHON_INSTALL_DIR),
        ],
        log,
    )


def ensure_venv(python_version: str, log: callable) -> None:
    if VENV_PYTHON.exists():
        log(f"Virtual environment already present: {VENV_PYTHON.relative_to(ROOT)}")
        return

    log("Creating .venv...")
    run_command(
        [
            str(UV_EXE),
            "venv",
            str(ROOT / ".venv"),
            "--python",
            python_version,
            "--managed-python",
        ],
        log,
    )

    if not VENV_PYTHON.exists():
        raise RuntimeError(".venv was created, but .venv\\Scripts\\python.exe is missing.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_requirements(requirements_path: str, log: callable) -> None:
    requirements = ROOT / requirements_path
    if not requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    current_hash = file_sha256(requirements)
    if REQUIREMENTS_STAMP.exists() and REQUIREMENTS_STAMP.read_text(encoding="utf-8") == current_hash:
        log("Python dependencies already match requirements.txt.")
        return

    log("Installing Python dependencies...")
    run_command(
        [
            str(UV_EXE),
            "pip",
            "install",
            "--python",
            str(VENV_PYTHON),
            "-r",
            str(requirements),
        ],
        log,
    )
    REQUIREMENTS_STAMP.parent.mkdir(parents=True, exist_ok=True)
    REQUIREMENTS_STAMP.write_text(current_hash, encoding="utf-8")


def launch_runtime(runtime_script: str, log: callable) -> None:
    script = ROOT / runtime_script
    if not script.exists():
        raise FileNotFoundError(f"Runtime script not found: {runtime_script}")
    log(f"Launching runtime menu: {runtime_script}")
    subprocess.Popen(
        [str(VENV_PYTHON), str(script)],
        cwd=ROOT,
        env=bootstrap_env(),
    )


def open_log_file() -> None:
    try:
        os.startfile(LOG_PATH)
    except Exception:
        print(f"Check log: {LOG_PATH}")


if __name__ == "__main__":
    raise SystemExit(BootstrapRunner().run())
