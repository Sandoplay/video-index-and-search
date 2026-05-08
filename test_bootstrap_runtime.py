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
    assert env["BOOTSTRAP_RUNTIME_EVENTS"] == "1"
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


def test_bootstrap_events_are_parsed_and_hidden_from_visible_log() -> None:
    root = reset_root()

    class StubProcess:
        def __init__(self) -> None:
            self.stdout = [
                '@@BOOTSTRAP_EVENT@@ {"kind":"discovery_summary","payload":{"new":1,"exact_matches":2,"conflicts":1,"items":[{"filename":"clip.mp4","action":"processed separately","input_path":"G:/input/clip.mp4","existing_path":"G:/done/clip.mp4"}]}}\n',
                "regular log line\n",
            ]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

    original_popen = bootstrap_runtime.subprocess.Popen
    bootstrap_runtime.subprocess.Popen = lambda *args, **kwargs: StubProcess()
    try:
        with patched_log_path(root):
            state = bootstrap_runtime.RuntimeState(bootstrap_runtime.DEFAULT_CONFIG)
            state.run_subprocess(["python", "-u", "main.py"], "index")
    finally:
        bootstrap_runtime.subprocess.Popen = original_popen

    assert state.discovery["new"] == 1
    assert state.discovery["exact_matches"] == 2
    assert state.discovery["conflicts"] == 1
    assert state.discovery["items"][0]["filename"] == "clip.mp4"
    assert not any("@@BOOTSTRAP_EVENT@@" in line for line in state.lines)
    assert any("regular log line" in line for line in state.lines)


def test_index_start_appends_reindex_flag_when_enabled() -> None:
    captured: dict[str, object] = {}

    class ImmediateThread:
        def __init__(self, *, target, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self) -> None:
            self._target(*self._args)

    state = bootstrap_runtime.RuntimeState(bootstrap_runtime.DEFAULT_CONFIG)
    state.reindex_existing = True

    original_thread = bootstrap_runtime.threading.Thread
    original_run_subprocess = state.run_subprocess
    bootstrap_runtime.threading.Thread = ImmediateThread
    state.run_subprocess = lambda args, name: captured.update({"args": args, "name": name})
    try:
        state.start("index", {"input": "G:/videos"})
    finally:
        bootstrap_runtime.threading.Thread = original_thread
        state.run_subprocess = original_run_subprocess

    assert captured["name"] == "index"
    assert captured["args"][-1] == "--reindex_existing"


def test_render_page_preserves_input_across_forms_and_restores_log_scroll() -> None:
    state = bootstrap_runtime.RuntimeState(bootstrap_runtime.DEFAULT_CONFIG)
    state.input_path = "G:/llm-video-test"
    handler = object.__new__(bootstrap_runtime.RuntimeHandler)
    handler.state = state

    page = handler.render_page()

    assert '<form id="setup-form" method="post" action="/run/setup">' in page
    assert '<form id="stop-form" method="post" action="/stop">' in page
    assert 'name="input" value="G:/llm-video-test"' in page
    assert 'const LOG_SCROLL_KEY = "bootstrap-runtime-log-scroll";' in page
    assert 'const PAGE_SCROLL_KEY = "bootstrap-runtime-page-scroll";' in page
    assert 'restorePageScrollState();' in page
    assert 'restoreLogScrollState();' in page


def main() -> None:
    tests = [
        test_build_command_args_uses_unbuffered_python_for_scripts,
        test_runtime_env_enables_unbuffered_python,
        test_run_subprocess_uses_line_buffered_text_mode,
        test_bootstrap_events_are_parsed_and_hidden_from_visible_log,
        test_index_start_appends_reindex_flag_when_enabled,
        test_render_page_preserves_input_across_forms_and_restores_log_scroll,
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
