import argparse
from src.search_service import SearchService, format_timestamp

def search_videos(query: str, limit: int = 5, item_type: str = None):
    """
    Search for videos or frames matching the given query using LanceDB vector search.
    """
    item_type = item_type or "summary"
    try:
        service = SearchService()
        results = service.search(query, item_type=item_type, limit=limit)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    if not results:
        print("No matching results found.")
        return
        
    print("\n" + "="*80)
    print(f"Top {len(results)} Results:")
    print("="*80)
    
    for i, res in enumerate(results, 1):
        score = res.get('distance', 0.0)
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
    parser.add_argument("--type", type=str, choices=['summary', 'all', 'segment'], default="summary", help="Filter results by type")
    
    args = parser.parse_args()
    search_videos(args.query, limit=args.limit, item_type=args.type)
