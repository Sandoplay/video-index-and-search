from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.indexer import VideoIndexer


TEST_ROOT = Path("data/temp_indexer_alias_cleanup_tests")


class StubTable:
    def __init__(self) -> None:
        self.deleted_filters: list[str] = []
        self.added_records: list[list[dict]] = []

    def delete(self, where: str) -> None:
        self.deleted_filters.append(where)

    def add(self, records: list[dict]) -> None:
        self.added_records.append(records)


def reset_root() -> Path:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_ROOT


def make_indexer() -> tuple[VideoIndexer, StubTable]:
    indexer = object.__new__(VideoIndexer)
    table = StubTable()
    indexer.table = table
    indexer._attach_embeddings = lambda records: [record.update({"vector": [0.0]}) for record in records]
    return indexer, table


def test_index_json_deletes_all_known_video_path_aliases() -> None:
    root = reset_root()
    json_path = root / "analysis_result.json"
    current_video = root / "indexed-videos" / "clip.mp4"
    source_video = root / "input" / "clip.mp4"
    old_archived_video = root / "completed" / "clip-old.mp4"
    json_path.write_text(
        json.dumps(
            {
                "video_path": str(current_video),
                "summary": "summary",
                "segments": [{"start_frame": 0, "end_frame": 0, "start_time": 0.0, "end_time": 1.0, "description": "segment"}],
                "analysis": [],
                "timings": {
                    "source_video_path": str(source_video),
                    "archived_video_path": str(current_video),
                    "indexed_video_path_aliases": [str(old_archived_video)],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    indexer, table = make_indexer()
    indexer.index_json(json_path)

    assert table.deleted_filters == [
        f"video_path = '{str(current_video)}'",
        f"video_path = '{str(source_video)}'",
        f"video_path = '{str(old_archived_video)}'",
    ]
    assert len(table.added_records) == 1
    assert len(table.added_records[0]) == 2


def main() -> None:
    tests = [
        test_index_json_deletes_all_known_video_path_aliases,
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

    print(f"\nAll {len(tests)} indexer alias cleanup tests passed.")


if __name__ == "__main__":
    main()
