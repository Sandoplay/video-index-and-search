import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from src.search_service import SearchService

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Video Semantic Search")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
SEARCH_STARTING_MESSAGE = "Search service is starting. Loading the embedding backend can take a while."
SEARCH_READY_MESSAGE = "Search service is ready."
ALLOWED_MEDIA_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".m4v",
}


def resolve_project_file(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")

    if resolved.suffix.lower() not in ALLOWED_MEDIA_SUFFIXES:
        raise HTTPException(status_code=403, detail="Unsupported media file type.")

    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        return resolved

    service = getattr(app.state, "search_service", None)
    library = getattr(service, "library", None)
    if library is not None and library.allows_media_path(resolved):
        return resolved

    raise HTTPException(status_code=403, detail="Path is outside the allowed media roots.")

    return resolved


def reveal_in_explorer(path: Path) -> None:
    normalized_path = str(path.resolve())
    subprocess.Popen(["explorer.exe", "/select,", normalized_path])


@app.on_event("startup")
def start_search_service_loader() -> None:
    app.state.search_service = None
    app.state.search_startup_error = ""
    app.state.search_status = "starting"
    app.state.search_status_message = SEARCH_STARTING_MESSAGE
    app.state.search_started_at = time.time()
    app.state.search_ready_at = None
    app.state.search_lock = threading.Lock()

    thread = threading.Thread(target=load_search_service_in_background, daemon=True)
    app.state.search_loader_thread = thread
    thread.start()


def load_search_service_in_background() -> None:
    try:
        print(SEARCH_STARTING_MESSAGE, flush=True)
        service = SearchService()
        with app.state.search_lock:
            app.state.search_service = service
            app.state.search_startup_error = ""
            app.state.search_status = "ready"
            app.state.search_status_message = SEARCH_READY_MESSAGE
            app.state.search_ready_at = time.time()
        print(SEARCH_READY_MESSAGE, flush=True)
    except Exception as exc:
        message = f"Search service failed to start: {exc}"
        with app.state.search_lock:
            app.state.search_service = None
            app.state.search_startup_error = str(exc)
            app.state.search_status = "error"
            app.state.search_status_message = message
            app.state.search_ready_at = None
        print(message, flush=True)


@app.on_event("shutdown")
def close_search_service() -> None:
    service = getattr(app.state, "search_service", None)
    if service is not None:
        service.close()


def search_status_payload() -> dict[str, Any]:
    lock = getattr(app.state, "search_lock", None)
    if lock is None:
        return {
            "status": "starting",
            "ready": False,
            "message": SEARCH_STARTING_MESSAGE,
            "error": "",
            "elapsed_seconds": 0.0,
        }

    with lock:
        status = getattr(app.state, "search_status", "starting")
        message = getattr(app.state, "search_status_message", SEARCH_STARTING_MESSAGE)
        error = getattr(app.state, "search_startup_error", "")
        started_at = getattr(app.state, "search_started_at", None)

    elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
    return {
        "status": status,
        "ready": status == "ready",
        "message": message,
        "error": error,
        "elapsed_seconds": round(elapsed, 1),
    }


def get_ready_search_service() -> SearchService:
    payload = search_status_payload()
    service = getattr(app.state, "search_service", None)
    if service is not None:
        return service

    if payload["status"] == "starting":
        raise HTTPException(status_code=503, detail=payload["message"])

    detail = payload["error"] or payload["message"] or "Search service is not available."
    raise HTTPException(status_code=503, detail=detail)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "service_status": search_status_payload(),
        },
    )


@app.get("/api/service-status")
def api_service_status() -> dict[str, Any]:
    return search_status_payload()


@app.get("/api/search")
def api_search(
    q: str = Query("", max_length=500),
    type: Literal["summary", "all", "segment", "transcription"] = "summary",
    limit: int = Query(10, ge=1),
    lexical_rerank: bool = Query(False),
) -> dict:
    query = q.strip()
    if not query:
        return {"query": query, "type": type, "limit": limit, "results": []}

    service = get_ready_search_service()

    try:
        results = service.search(query, item_type=type, limit=limit, lexical_rerank=lexical_rerank)
    except Exception as exc:
        print(f"Search failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"query": query, "type": type, "limit": limit, "lexical_rerank": lexical_rerank, "results": results}


@app.get("/media")
def local_media(path: str) -> FileResponse:
    resolved = resolve_project_file(path)
    return FileResponse(str(resolved))


@app.post("/api/reveal-file")
def reveal_file(path: str) -> dict:
    resolved = resolve_project_file(path)
    reveal_in_explorer(resolved)
    return {"ok": True, "path": str(resolved)}


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
