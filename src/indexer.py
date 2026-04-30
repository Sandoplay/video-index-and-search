import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import lancedb
from lancedb.pydantic import Vector, LanceModel

from src.embedding_provider import (
    DB_PATH,
    MINILM_MODEL_NAME,
    MINILM_TABLE_NAME,
    MINILM_VECTOR_DIM,
    MODEL_CACHE_DIR,
    create_embedding_provider,
)

MODEL_NAME = MINILM_MODEL_NAME
TABLE_NAME = MINILM_TABLE_NAME
VECTOR_DIM = MINILM_VECTOR_DIM


class VideoRecord(LanceModel):
    id: str
    video_path: str
    timestamp: float
    text: str
    type: str # 'summary', 'segment', or 'frame'
    vector: Vector(VECTOR_DIM) # type: ignore


def create_video_record_schema(vector_dim: int):
    class DynamicVideoRecord(LanceModel):
        id: str
        video_path: str
        timestamp: float
        text: str
        type: str
        vector: Vector(vector_dim) # type: ignore

    return DynamicVideoRecord


class VideoIndexer:
    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        model_name: str = MODEL_NAME,
        cache_dir: str | Path = MODEL_CACHE_DIR,
        embedding_backend: str | None = None,
        table_name: str | None = None,
        embedding_context_size: int | None = None,
        embedding_parallel: int | None = None,
    ):
        self.db_path = str(db_path)
        self.model_name = model_name
        self.cache_dir = str(cache_dir)
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.provider = create_embedding_provider(
            backend=embedding_backend,
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            context_size=embedding_context_size,
            parallel_slots=embedding_parallel,
        )
        self.table_name = table_name or os.getenv("EMBEDDING_TABLE_NAME") or self.provider.table_name
        self.schema = create_video_record_schema(self.provider.vector_dim)
        
        # Connect to LanceDB
        print(f"Connecting to LanceDB at: {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        
        # Open or create the table
        tables = self.db.list_tables()
        table_names = tables.tables if hasattr(tables, "tables") else tables
        if self.table_name in table_names:
            self.table = self.db.open_table(self.table_name)
        else:
            self.table = self.db.create_table(self.table_name, schema=self.schema)
            
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a dense vector embedding for the given text."""
        return self.provider.embed_document(text)

    def _attach_embeddings(self, records: List[Dict[str, Any]]) -> None:
        texts = [str(record.get("text") or "") for record in records]
        vectors = self.provider.embed_documents(texts)
        for record, vector in zip(records, vectors):
            record["vector"] = vector

    def close(self) -> None:
        self.provider.close()

    def _records_from_json(self, json_path: str | Path) -> Tuple[str, List[Dict[str, Any]]]:
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"File not found: {json_path}")
            return "", []

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        video_path = data.get("video_path", "")
        if not video_path:
            video_path = str(json_file.parent.name) # fallback

        records: List[Dict[str, Any]] = []

        summary_text = data.get("summary", "")
        if summary_text:
            summary_id = f"{video_path}_summary"
            records.append({
                "id": summary_id,
                "video_path": video_path,
                "timestamp": 0.0,
                "text": summary_text,
                "type": "summary",
            })

        segments = data.get("segments", [])
        if segments:
            for i, segment in enumerate(segments):
                timestamp = segment.get("start_time", 0.0)
                desc = segment.get("description", "")
                if desc:
                    segment_id = f"{video_path}_segment_{i}_{timestamp}"
                    records.append({
                        "id": segment_id,
                        "video_path": video_path,
                        "timestamp": float(timestamp),
                        "text": desc,
                        "type": "segment",
                    })
        else:
            analysis = data.get("analysis", [])
            if analysis:
                for i, frame in enumerate(analysis):
                    timestamp = frame.get("timestamp", 0.0)
                    desc = frame.get("description", "")
                    if desc:
                        frame_id = f"{video_path}_frame_{i}_{timestamp}"
                        records.append({
                            "id": frame_id,
                            "video_path": video_path,
                            "timestamp": float(timestamp),
                            "text": desc,
                            "type": "frame",
                        })

        return video_path, records
        
    def index_json(self, json_path: str | Path):
        """Parse an analysis_result.json file and insert its contents into the database."""
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"File not found: {json_path}")
            return
            
        print(f"Indexing: {json_file.name}...")
        video_path, records = self._records_from_json(json_file)
                    
        if records:
            print(f"  -> Generating {len(records)} embeddings...")
            self._attach_embeddings(records)

            # We can use merge_insert to handle duplicates (upsert based on id)
            # But simple insert is fine for now, let's just insert.
            # To be safer about re-runs, we delete existing records for this video first.
            try:
                # LanceDB SQL filters match Windows backslashes literally.
                # Escaping them as "\\" prevents old rows from being deleted.
                safe_video_path = video_path.replace("'", "''")
                self.table.delete(f"video_path = '{safe_video_path}'")
                print(f"  -> Cleared old records for {video_path}")
            except Exception as e:
                print(f"  -> Warning: Could not delete old records (might be empty): {e}")
                
            print(f"  -> Inserting {len(records)} records into LanceDB...")
            self.table.add(records)
            print(f"Successfully indexed {json_file.name}.")
        else:
            print("No indexable text found in JSON.")

    def index_json_files(
        self,
        json_paths: List[str | Path],
        embedding_batch_size: int = 32,
        add_batch_size: int = 512,
    ) -> None:
        parsed: List[Tuple[str, List[Dict[str, Any]]]] = []
        all_records: List[Dict[str, Any]] = []

        for json_path in json_paths:
            video_path, records = self._records_from_json(json_path)
            if not video_path or not records:
                continue
            parsed.append((video_path, records))
            all_records.extend(records)

        if not all_records:
            print("No indexable text found in JSON files.")
            return

        print(f"Preparing {len(all_records)} record(s) from {len(parsed)} video(s).")
        embedding_batch_size = max(1, int(embedding_batch_size))
        for start in range(0, len(all_records), embedding_batch_size):
            batch = all_records[start:start + embedding_batch_size]
            end = start + len(batch)
            print(f"  -> Generating embeddings {start + 1}-{end}/{len(all_records)}...")
            self._attach_embeddings(batch)

        for video_path, _ in parsed:
            try:
                safe_video_path = video_path.replace("'", "''")
                self.table.delete(f"video_path = '{safe_video_path}'")
            except Exception as e:
                print(f"  -> Warning: Could not delete old records for {video_path}: {e}")

        add_batch_size = max(1, int(add_batch_size))
        for start in range(0, len(all_records), add_batch_size):
            batch = all_records[start:start + add_batch_size]
            end = start + len(batch)
            print(f"  -> Inserting records {start + 1}-{end}/{len(all_records)}...")
            self.table.add(batch)

        print(f"Successfully indexed {len(parsed)} video(s).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index video metadata JSON into LanceDB")
    parser.add_argument("json_path", type=str, help="Path to analysis_result.json to index")
    parser.add_argument("--embedding-backend", choices=["minilm", "qwen3-gguf", "qwen3-4b-gguf"], default=None)
    parser.add_argument("--table", default=None, help="Override LanceDB table name")
    parser.add_argument("--embedding-ctx-size", type=int, default=None)
    parser.add_argument("--embedding-parallel", type=int, default=None)
    args = parser.parse_args()
    
    indexer = VideoIndexer(
        embedding_backend=args.embedding_backend,
        table_name=args.table,
        embedding_context_size=args.embedding_ctx_size,
        embedding_parallel=args.embedding_parallel,
    )
    try:
        indexer.index_json(args.json_path)
    finally:
        indexer.close()
