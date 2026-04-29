import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import lancedb

os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/lancedb")
MODEL_CACHE_DIR = Path("models/.cache/huggingface")
MODEL_NAME = "all-MiniLM-L6-v2"
TABLE_NAME = "video_metadata"


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def _table_names(db: Any) -> List[str]:
    tables = db.list_tables()
    return list(tables.tables if hasattr(tables, "tables") else tables)


def _normalize_path_key(path_value: str) -> str:
    return os.path.normcase(os.path.normpath(path_value.replace("/", os.sep)))


def _absolute_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(path.resolve())


class AnalysisLibrary:
    def __init__(self, data_root: str | Path = "data"):
        self.data_root = Path(data_root)
        self.by_video_path: Dict[str, Dict[str, Any]] = {}
        self.scan()

    def scan(self) -> None:
        candidates: Dict[str, Dict[str, Any]] = {}
        if not self.data_root.exists():
            self.by_video_path = {}
            return

        roots = [
            path for path in self.data_root.glob("processed*")
            if path.is_dir()
        ]
        for root in roots:
            for json_path in root.rglob("analysis_result.json"):
                try:
                    with json_path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue

                video_path = str(data.get("video_path") or "")
                if not video_path:
                    continue

                entry = {
                    "json_path": str(json_path),
                    "json_mtime": json_path.stat().st_mtime,
                    "data": data,
                }

                for key in {video_path, _normalize_path_key(video_path)}:
                    existing = candidates.get(key)
                    if existing is None or entry["json_mtime"] > existing["json_mtime"]:
                        candidates[key] = entry

        self.by_video_path = candidates

    def get(self, video_path: str) -> Optional[Dict[str, Any]]:
        return self.by_video_path.get(video_path) or self.by_video_path.get(_normalize_path_key(video_path))

    def enrich(self, result: Dict[str, Any]) -> Dict[str, Any]:
        video_path = str(result.get("video_path") or "")
        entry = self.get(video_path)
        if not entry:
            return result

        data = entry["data"]
        result["source_json"] = entry["json_path"]
        result["duration"] = data.get("duration")
        result["summary"] = data.get("summary") or ""

        preview_path = self._preview_path(data, result, Path(entry["json_path"]).parent)
        if preview_path:
            result["preview_path"] = preview_path
            result["preview_url"] = f"/media?path={quote(preview_path)}"

        return result

    def _preview_path(self, data: Dict[str, Any], result: Dict[str, Any], json_dir: Path) -> Optional[str]:
        analysis = data.get("analysis") or []
        timestamp = float(result.get("timestamp") or 0.0)

        if analysis:
            nearest = min(
                analysis,
                key=lambda item: abs(float(item.get("timestamp") or 0.0) - timestamp),
            )
            frame_path = nearest.get("frame_path")
            if frame_path:
                return str(frame_path)

        segments = data.get("segments") or []
        if segments:
            nearest_segment = min(
                segments,
                key=lambda item: abs(float(item.get("start_time") or 0.0) - timestamp),
            )
            start_frame = int(nearest_segment.get("start_frame") or 0)
            frame_path = json_dir / "frames" / f"frame_{start_frame + 1:04d}.jpg"
            if frame_path.exists():
                return str(frame_path)

        return None


class SearchService:
    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        model_name: str = MODEL_NAME,
        cache_dir: str | Path = MODEL_CACHE_DIR,
        table_name: str = TABLE_NAME,
    ):
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.table_name = table_name

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run the indexer first.")

        print(f"Loading embedding model once: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))

        print(f"Connecting to LanceDB at: {self.db_path}")
        self.db = lancedb.connect(str(self.db_path))
        if self.table_name not in _table_names(self.db):
            raise RuntimeError(f"Table '{self.table_name}' not found in {self.db_path}. Run the indexer first.")

        self.table = self.db.open_table(self.table_name)
        self.library = AnalysisLibrary()

    def search(self, query: str, item_type: str = "summary", limit: int = 10) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        if item_type not in {"summary", "all", "segment"}:
            raise ValueError("item_type must be one of: summary, all, segment")

        limit = max(1, min(int(limit), 100))
        candidate_limit = min(max(limit * 8, limit), 500)
        query_vector = self.model.encode(query).tolist()
        search = self.table.search(query_vector).limit(candidate_limit)

        if item_type in {"summary", "segment"}:
            search = search.where(f"type = '{item_type}'")

        deduped_rows = self._dedupe_rows(search.to_list(), limit)
        return [
            self.library.enrich(self._normalize_result(index, row))
            for index, row in enumerate(deduped_rows, 1)
        ]

    def _dedupe_rows(self, rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        for row in rows:
            key = self._dedupe_key(row)
            if key in seen:
                continue

            seen.add(key)
            results.append(row)
            if len(results) >= limit:
                break

        return results

    def _dedupe_key(self, row: Dict[str, Any]) -> tuple[Any, ...]:
        item_type = str(row.get("type") or "")
        video_path = _normalize_path_key(str(row.get("video_path") or ""))

        if item_type == "summary":
            return (item_type, video_path)

        timestamp = round(float(row.get("timestamp") or 0.0), 3)
        return (item_type, video_path, timestamp)

    def _normalize_result(self, rank: int, row: Dict[str, Any]) -> Dict[str, Any]:
        video_path = str(row.get("video_path") or "")
        timestamp = float(row.get("timestamp") or 0.0)
        item_type = str(row.get("type") or "")
        return {
            "rank": rank,
            "id": row.get("id") or "",
            "video_path": video_path,
            "video_abs_path": _absolute_path(video_path) if video_path else "",
            "video_name": Path(video_path).name or video_path,
            "type": item_type,
            "timestamp": timestamp,
            "timestamp_label": format_timestamp(timestamp) if item_type in {"segment", "frame"} else "",
            "text": row.get("text") or "",
            "distance": float(row.get("_distance") or 0.0),
            "preview_path": "",
            "preview_url": "",
            "source_json": "",
            "duration": None,
            "summary": "",
        }
