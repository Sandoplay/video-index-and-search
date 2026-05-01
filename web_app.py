import subprocess
from pathlib import Path
from typing import Literal

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


def resolve_project_file(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    resolved = candidate.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise HTTPException(status_code=403, detail="Path is outside the project directory.")

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")

    return resolved


@app.on_event("startup")
def load_search_service() -> None:
    try:
        app.state.search_service = SearchService()
        app.state.search_startup_error = ""
    except Exception as exc:
        app.state.search_service = None
        app.state.search_startup_error = str(exc)
        print(f"Search service failed to start: {exc}")


@app.on_event("shutdown")
def close_search_service() -> None:
    service = getattr(app.state, "search_service", None)
    if service is not None:
        service.close()


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "startup_error": getattr(app.state, "search_startup_error", ""),
        },
    )


@app.get("/api/search")
def api_search(
    q: str = Query("", max_length=500),
    type: Literal["summary", "all", "segment"] = "summary",
    limit: int = Query(10, ge=1),
    lexical_rerank: bool = Query(False),
) -> dict:
    query = q.strip()
    if not query:
        return {"query": query, "type": type, "limit": limit, "results": []}

    service = getattr(app.state, "search_service", None)
    if service is None:
        detail = getattr(app.state, "search_startup_error", "") or "Search service is not available."
        raise HTTPException(status_code=503, detail=detail)

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
    subprocess.Popen(["explorer.exe", f"/select,{resolved}"])
    return {"ok": True, "path": str(resolved)}


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
