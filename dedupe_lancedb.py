import argparse
from pathlib import Path
from typing import Any, Dict

import lancedb

from src.indexer import DB_PATH, TABLE_NAME, VideoRecord


def dedupe_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    item_type = str(row.get("type") or "")
    video_path = str(row.get("video_path") or "")

    if item_type == "summary":
        return (item_type, video_path)

    timestamp = round(float(row.get("timestamp") or 0.0), 3)
    return (item_type, video_path, timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or compact duplicate rows in the LanceDB video metadata table.")
    parser.add_argument("--db-path", default=str(DB_PATH), help="Path to LanceDB directory")
    parser.add_argument("--table", default=TABLE_NAME, help="LanceDB table name")
    parser.add_argument("--apply", action="store_true", help="Rewrite the table with duplicate rows removed")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db = lancedb.connect(str(db_path))
    tables = db.list_tables()
    table_names = tables.tables if hasattr(tables, "tables") else tables
    if args.table not in table_names:
        raise SystemExit(f"Table '{args.table}' not found in {db_path}.")

    table = db.open_table(args.table)
    rows = table.to_arrow().to_pylist()

    deduped_by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        deduped_by_key[dedupe_key(row)] = row

    deduped_rows = list(deduped_by_key.values())
    removed = len(rows) - len(deduped_rows)

    print(f"Rows before: {len(rows)}")
    print(f"Rows after:  {len(deduped_rows)}")
    print(f"Duplicates:  {removed}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to rewrite the table.")
        return

    db.drop_table(args.table)
    db.create_table(args.table, data=deduped_rows, schema=VideoRecord)
    print(f"Rewrote table '{args.table}' with duplicates removed.")


if __name__ == "__main__":
    main()
