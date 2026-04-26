import argparse
import os
from pathlib import Path
import lancedb

# Prevent huggingface_hub from making network requests
os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

# Define constants
DB_PATH = Path("data/lancedb")
MODEL_CACHE_DIR = Path("models/.cache/huggingface")
MODEL_NAME = "all-MiniLM-L6-v2"
TABLE_NAME = "video_metadata"

def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.sss"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"

def search_videos(query: str, limit: int = 5, item_type: str = None):
    """
    Search for videos or frames matching the given query using LanceDB vector search.
    """
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}. Please run indexer first.")
        return

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    
    print(f"Connecting to LanceDB at: {DB_PATH}")
    db = lancedb.connect(str(DB_PATH))
    
    if TABLE_NAME not in db.list_tables().tables:
        print(f"Error: Table '{TABLE_NAME}' not found in database. Please run indexer first.")
        return
        
    table = db.open_table(TABLE_NAME)
    
    print(f"\nSearching for: '{query}'...")
    query_vector = model.encode(query).tolist()
    
    # Perform vector search
    search = table.search(query_vector).limit(limit)
    
    # Filter by type if provided (e.g., 'summary' or 'frame')
    if item_type:
        search = search.where(f"type = '{item_type}'")
        
    results = search.to_list()
    
    if not results:
        print("No matching results found.")
        return
        
    print("\n" + "="*80)
    print(f"Top {len(results)} Results:")
    print("="*80)
    
    for i, res in enumerate(results, 1):
        score = res.get('_distance', 0.0)
        v_path = res.get('video_path', 'Unknown')
        t_type = res.get('type', 'Unknown')
        t_stamp = res.get('timestamp', 0.0)
        text = res.get('text', '')
        
        # Truncate text for display
        display_text = text if len(text) < 150 else text[:147] + "..."
        
        print(f"\nResult {i} (Distance: {score:.4f})")
        print(f"Video:     {v_path}")
        print(f"Type:      {t_type.capitalize()}")
        if t_type in ("frame", "segment"):
            print(f"Timestamp: {format_timestamp(t_stamp)} ({t_stamp:.2f}s)")
        print(f"Text:      {display_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search for Indexed Videos")
    parser.add_argument("query", type=str, help="Natural language query to search for")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return (default: 5)")
    parser.add_argument("--type", type=str, choices=['summary', 'segment', 'frame'], help="Filter results by type (summary, segment, or frame)")
    
    args = parser.parse_args()
    search_videos(args.query, limit=args.limit, item_type=args.type)
