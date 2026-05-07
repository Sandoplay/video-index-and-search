from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

import web_app


TEST_ROOT = Path("data/temp_web_app_path_tests")


class StubLibrary:
    def __init__(self, allowed_paths: set[str]):
        self.allowed_paths = {str(Path(path).resolve()) for path in allowed_paths}

    def allows_media_path(self, path: str | Path) -> bool:
        return str(Path(path).resolve()) in self.allowed_paths


@contextmanager
def patched_project_root(path: Path):
    original = web_app.PROJECT_ROOT
    web_app.PROJECT_ROOT = path.resolve()
    try:
        yield
    finally:
        web_app.PROJECT_ROOT = original


def reset_root() -> Path:
    import shutil

    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_ROOT


def test_project_media_file_is_allowed() -> None:
    root = reset_root()
    project_root = root / "project"
    static_dir = project_root / "data"
    static_dir.mkdir(parents=True, exist_ok=True)
    media_file = static_dir / "preview.jpg"
    media_file.write_bytes(b"jpg")

    web_app.app.state.search_service = None
    with patched_project_root(project_root):
        resolved = web_app.resolve_project_file(str(media_file.resolve()))

    assert resolved == media_file.resolve()


def test_known_external_indexed_video_is_allowed() -> None:
    root = reset_root()
    project_root = root / "project"
    external_dir = root / "external" / "indexed-videos"
    external_dir.mkdir(parents=True, exist_ok=True)
    media_file = external_dir / "clip.mp4"
    media_file.write_bytes(b"mp4")

    web_app.app.state.search_service = SimpleNamespace(
        library=StubLibrary({str(media_file)})
    )
    with patched_project_root(project_root):
        resolved = web_app.resolve_project_file(str(media_file.resolve()))

    assert resolved == media_file.resolve()


def test_unknown_external_file_is_blocked() -> None:
    root = reset_root()
    project_root = root / "project"
    external_dir = root / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    media_file = external_dir / "secret.mp4"
    media_file.write_bytes(b"mp4")

    web_app.app.state.search_service = SimpleNamespace(
        library=StubLibrary(set())
    )
    with patched_project_root(project_root):
        try:
            web_app.resolve_project_file(str(media_file.resolve()))
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("Unknown external media file should be blocked")


def test_reveal_in_explorer_uses_select_argument_and_absolute_path() -> None:
    root = reset_root()
    media_file = (root / "video.mp4").resolve()
    media_file.write_bytes(b"mp4")
    recorded = {}

    class StubPopen:
        def __init__(self, args):
            recorded["args"] = args

    original_popen = web_app.subprocess.Popen
    web_app.subprocess.Popen = StubPopen
    try:
        web_app.reveal_in_explorer(media_file)
    finally:
        web_app.subprocess.Popen = original_popen

    assert recorded["args"] == ["explorer.exe", "/select,", str(media_file)]


def main() -> None:
    tests = [
        test_project_media_file_is_allowed,
        test_known_external_indexed_video_is_allowed,
        test_unknown_external_file_is_blocked,
        test_reveal_in_explorer_uses_select_argument_and_absolute_path,
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

    print(f"\nAll {len(tests)} web app path tests passed.")


if __name__ == "__main__":
    main()
