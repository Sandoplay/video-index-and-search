import json
import os
from pathlib import Path
from typing import List, Dict, Any
import lancedb
from lancedb.pydantic import Vector, LanceModel

# Prevent huggingface_hub from making network requests
os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

# Define constants
DB_PATH = Path("data/lancedb")
MODEL_CACHE_DIR = Path("models/.cache/huggingface")
MODEL_NAME = "all-MiniLM-L6-v2"
TABLE_NAME = "video_metadata"

# Ensure directories exist
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Determine the vector dimensionality
# all-MiniLM-L6-v2 produces 384-dimensional vectors
VECTOR_DIM = 384

# Define the LanceDB schema
class VideoRecord(LanceModel):
    id: str
    video_path: str
    timestamp: float
    text: str
    type: str # 'summary', 'segment', or 'frame'
    vector: Vector(VECTOR_DIM) # type: ignore

class VideoIndexer:
    def __init__(self, db_path: str | Path = DB_PATH, model_name: str = MODEL_NAME, cache_dir: str | Path = MODEL_CACHE_DIR):
        self.db_path = str(db_path)
        self.model_name = model_name
        self.cache_dir = str(cache_dir)
        
        # Load embedding model
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
        
        # Connect to LanceDB
        print(f"Connecting to LanceDB at: {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        
        # Open or create the table
        if TABLE_NAME in self.db.list_tables().tables:
            self.table = self.db.open_table(TABLE_NAME)
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=VideoRecord)
            
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a dense vector embedding for the given text."""
        return self.model.encode(text).tolist()
        
    def index_json(self, json_path: str | Path):
        """Parse an analysis_result.json file and insert its contents into the database."""
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"File not found: {json_path}")
            return
            
        print(f"Indexing: {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        video_path = data.get("video_path", "")
        if not video_path:
            video_path = str(json_file.parent.name) # fallback
            
        records: List[Dict[str, Any]] = []
        
        # 1. Index the overall summary
        summary_text = data.get("summary", "")
        if summary_text:
            print("  -> Indexing summary...")
            summary_id = f"{video_path}_summary"
            records.append({
                "id": summary_id,
                "video_path": video_path,
                "timestamp": 0.0,
                "text": summary_text,
                "type": "summary",
                "vector": self._generate_embedding(summary_text)
            })
            
        # 2. Prefer structured timeline segments; fall back to legacy frame descriptions.
        segments = data.get("segments", [])
        if segments:
            print(f"  -> Indexing {len(segments)} timeline segments...")
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
                        "vector": self._generate_embedding(desc)
                    })
        else:
            analysis = data.get("analysis", [])
            if analysis:
                print(f"  -> Indexing {len(analysis)} frame descriptions...")
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
                            "vector": self._generate_embedding(desc)
                        })
                    
        if records:
            # We can use merge_insert to handle duplicates (upsert based on id)
            # But simple insert is fine for now, let's just insert.
            # To be safer about re-runs, we delete existing records for this video first.
            try:
                # Екрануємо зворотні слеші та одинарні лапки для SQL-фільтра
                safe_video_path = video_path.replace("\\", "\\\\").replace("'", "''")
                self.table.delete(f"video_path = '{safe_video_path}'")
                print(f"  -> Cleared old records for {video_path}")
            except Exception as e:
                print(f"  -> Warning: Could not delete old records (might be empty): {e}")
                
            print(f"  -> Inserting {len(records)} records into LanceDB...")
            self.table.add(records)
            print(f"Successfully indexed {json_file.name}.")
        else:
            print("No indexable text found in JSON.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index video metadata JSON into LanceDB")
    parser.add_argument("json_path", type=str, help="Path to analysis_result.json to index")
    args = parser.parse_args()
    
    indexer = VideoIndexer()
    indexer.index_json(args.json_path)
