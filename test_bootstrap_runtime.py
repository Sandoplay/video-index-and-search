from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import bootstrap_runtime


TEST_ROOT = Path("data/temp_bootstrap_runtime_tests")


@contextmanager
def patched_log_path(root: Path):
    original_log_dir = bootstrap_runtime.LOG_DIR
    original_log_path = bootstrap_runtime.LOG_PATH
    bootstrap_runtime.LOG_DIR = root.resolve()
    bootstrap_runtime.LOG_PATH = bootstrap_runtime.LOG_DIR / "bootstrap.log"
    try:
        yield
    finally:
        bootstrap_runtime.LOG_DIR = original_log_dir
        bootstrap_runtime.LOG_PATH = original_log_path


def reset_root() -> Path:
    import shutil

    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_ROOT


def test_build_command_args_uses_unbuffered_python_for_scripts() -> None:
    args = bootstrap_runtime.build_command_args(
        ["main.py", "--input", "{input}", "--embedding_backend", "{embedding_backend}"],
        {"input": "G:/videos", "embedding_backend": "qwen3-4b-gguf"},
    )

    assert args[0] == str(bootstrap_runtime.python_executable())
    assert args[1] == "-u"
    assert Path(args[2]) == (bootstrap_runtime.ROOT / "main.py")
    assert args[3:] == ["--input", "G:/videos", "--embedding_backend", "qwen3-4b-gguf"]


def test_runtime_env_enables_unbuffered_python() -> None:
    env = bootstrap_runtime.runtime_env()

    assert env["PYTHONUNBUFFERED"] == "1"
    assert str(bootstrap_runtime.ROOT / ".venv" / "Scripts") in env["PATH"]


def test_run_subprocess_uses_line_buffered_text_mode() -> None:
    root = reset_root()
    recorded: dict[str, object] = {}

    class StubProcess:
        def __init__(self) -> None:
            self.stdout = ["alpha\n", "beta\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

    def stub_popen(*args, **kwargs):
        recorded["kwargs"] = kwargs
        return StubProcess()

    original_popen = bootstrap_runtime.subprocess.Popen
    bootstrap_runtime.subprocess.Popen = stub_popen
    try:
        with patched_log_path(root):
            state = bootstrap_runtime.RuntimeState(bootstrap_runtime.DEFAULT_CONFIG)
            state.run_subprocess(["python", "-u", "main.py"], "index")
    finally:
        bootstrap_runtime.subprocess.Popen = original_popen

    kwargs = recorded["kwargs"]
    assert kwargs["bufsize"] == 1
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"


def main() -> None:
    tests = [
        test_build_command_args_uses_unbuffered_python_for_scripts,
        test_runtime_env_enables_unbuffered_python,
        test_run_subprocess_uses_line_buffered_text_mode,
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

    print(f"\nAll {len(tests)} bootstrap runtime tests passed.")


if __name__ == "__main__":
    main()
