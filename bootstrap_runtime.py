from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "bootstrap_config.json"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "bootstrap.log"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MENU_HOST = "127.0.0.1"
MENU_PORT = 8765
BOOTSTRAP_EVENT_PREFIX = "@@BOOTSTRAP_EVENT@@ "

DEFAULT_CONFIG = {
    "default_embedding_backend": "qwen3-4b-gguf",
    "commands": {
        "setup": ["setup_models.py", "--yes"],
        "web": ["web_app.py"],
        "index": [
            "main.py",
            "--input",
            "{input}",
            "--output_dir",
            "data/processed",
            "--max_frames",
            "50",
            "--batch_size",
            "0",
            "--parallel_requests",
            "2",
            "--server_parallel",
            "2",
            "--embedding_backend",
            "{embedding_backend}",
        ],
    }
}

EMBEDDING_BACKENDS = {
    "qwen3-4b-gguf": "Qwen3 4B GGUF",
    "qwen3-gguf": "Qwen3 0.6B GGUF",
    "minilm": "MiniLM",
}


def normalize_embedding_backend(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in EMBEDDING_BACKENDS else "qwen3-4b-gguf"


def build_command_args(command: list[str], replacements: dict[str, str] | None = None) -> list[str]:
    replacements = replacements or {}
    resolved = [part.format(**replacements) for part in command]
    entrypoint = resolved[0]
    entry_path = ROOT / entrypoint
    if entry_path.suffix.lower() == ".py":
        return [str(python_executable()), "-u", str(entry_path), *resolved[1:]]
    return [str(entry_path if entry_path.exists() else entrypoint), *resolved[1:]]


class RuntimeState:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_path = "data/input"
        self.embedding_backend = normalize_embedding_backend(
            str(config.get("default_embedding_backend") or "qwen3-4b-gguf")
        )
        self.reindex_existing = False
        self.lines: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.discovery = self.empty_discovery_state()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def empty_discovery_state() -> dict:
        return {
            "new": 0,
            "exact_matches": 0,
            "conflicts": 0,
            "items": [],
        }

    def reset_discovery(self) -> None:
        with self.lock:
            self.discovery = self.empty_discovery_state()

    def apply_bootstrap_event(self, payload: dict) -> None:
        kind = payload.get("kind")
        body = payload.get("payload") or {}
        with self.lock:
            if kind == "discovery_summary":
                self.discovery = {
                    "new": int(body.get("new") or 0),
                    "exact_matches": int(body.get("exact_matches") or 0),
                    "conflicts": int(body.get("conflicts") or 0),
                    "items": list(body.get("items") or []),
                }
            elif kind == "discovery_conflict":
                items = list(self.discovery.get("items") or [])
                if not any(
                    str(item.get("input_path") or "") == str(body.get("input_path") or "")
                    and str(item.get("existing_path") or "") == str(body.get("existing_path") or "")
                    for item in items
                ):
                    items.append(body)
                self.discovery["items"] = items
                self.discovery["conflicts"] = len(items)

    def handle_subprocess_line(self, line: str) -> bool:
        if not line.startswith(BOOTSTRAP_EVENT_PREFIX):
            return False
        try:
            payload = json.loads(line[len(BOOTSTRAP_EVENT_PREFIX):])
        except json.JSONDecodeError:
            return False
        self.apply_bootstrap_event(payload)
        return True

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.lock:
            self.lines.append(message)
            self.lines = self.lines[-700:]
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    def is_busy(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.is_busy() and self.process is not None:
            self.log("Stopping running command...")
            self.process.terminate()

    def start(self, name: str, replacements: dict[str, str] | None = None) -> None:
        if self.is_busy():
            self.log("A command is already running.")
            return

        command = self.config.get("commands", {}).get(name)
        if not command:
            self.log(f"Command is missing in bootstrap_config.json: {name}")
            return
        if name == "index":
            self.reset_discovery()

        args = build_command_args(
            command,
            {
                "embedding_backend": self.embedding_backend,
                **(replacements or {}),
            },
        )
        if name == "index" and self.reindex_existing:
            args.append("--reindex_existing")
        threading.Thread(target=self.run_subprocess, args=(args, name), daemon=True).start()

    def run_subprocess(self, args: list[str], name: str) -> None:
        try:
            self.log("> " + " ".join(args))
            env = runtime_env()
            if name == "web":
                env["EMBEDDING_BACKEND"] = self.embedding_backend
                self.log(f"Using embedding backend for web search: {self.embedding_backend}")
            elif name == "index":
                self.log(f"Using embedding backend for indexing: {self.embedding_backend}")

            self.process = subprocess.Popen(
                args,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if name == "web":
                threading.Thread(target=open_search_ui_after_delay, daemon=True).start()
            assert self.process.stdout is not None
            for line in self.process.stdout:
                stripped = line.rstrip()
                if self.handle_subprocess_line(stripped):
                    continue
                self.log(stripped)
            exit_code = self.process.wait()
            if exit_code == 0:
                self.log("Command finished successfully.")
            else:
                self.log(f"Command failed with exit code {exit_code}.")
        except Exception as exc:
            self.log(f"Command failed: {exc}")
        finally:
            self.process = None


class RuntimeHandler(BaseHTTPRequestHandler):
    state: RuntimeState

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self.send_json(self.status_payload())
            return
        self.send_html(self.render_page())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw_body)
        input_path = form.get("input", [self.state.input_path])[0].strip() or "data/input"
        self.state.input_path = input_path
        self.state.embedding_backend = normalize_embedding_backend(
            form.get("embedding_backend", [self.state.embedding_backend])[0]
        )
        self.state.reindex_existing = form.get("reindex_existing", [""])[0].lower() in {"1", "true", "on", "yes"}

        if self.path == "/run/setup":
            self.state.start("setup")
        elif self.path == "/run/web":
            self.state.start("web")
        elif self.path == "/run/index":
            self.state.start("index", {"input": input_path})
        elif self.path == "/stop":
            self.state.stop()

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def status_payload(self) -> dict:
        with self.state.lock:
            lines = list(self.state.lines)
            discovery = dict(self.state.discovery)
        busy = self.state.is_busy()
        return {
            "busy": busy,
            "status": "Running" if busy else "Ready",
            "input_path": self.state.input_path,
            "embedding_backend": self.state.embedding_backend,
            "reindex_existing": self.state.reindex_existing,
            "discovery": discovery,
            "log": "\n".join(lines),
        }

    def render_page(self) -> str:
        with self.state.lock:
            lines = "\n".join(escape(line) for line in self.state.lines)
        busy = self.state.is_busy()
        disabled = "disabled" if busy else ""
        stop_disabled = "" if busy else "disabled"
        input_value = escape(self.state.input_path, quote=True)
        embedding_value = self.state.embedding_backend
        reindex_checked = "checked" if self.state.reindex_existing else ""
        embedding_options = "\n".join(
            f'<option value="{escape(value, quote=True)}" {"selected" if value == embedding_value else ""}>{escape(label)}</option>'
            for value, label in EMBEDDING_BACKENDS.items()
        )
        status = "Running" if busy else "Ready"
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ml-vid-indexer-search</title>
  <script>
    (() => {{
      const saved = localStorage.getItem("video-search-theme");
      const theme = saved === "dark" || saved === "light"
        ? saved
        : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      document.documentElement.dataset.theme = theme;
    }})();
  </script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f4;
      --ink: #18201d;
      --muted: #63706b;
      --line: #d7ddd8;
      --panel: #ffffff;
      --panel-soft: #e9eeea;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warm: #b45309;
      --danger: #a32020;
      --focus: rgba(15, 118, 110, 0.14);
      --code: #111412;
      --code-text: #edf3ef;
      --shadow: rgba(24, 32, 29, 0.08);
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
      --bg: #111412;
      --ink: #edf3ef;
      --muted: #a7b4ae;
      --line: #303a35;
      --panel: #1a201d;
      --panel-soft: #202822;
      --accent: #2dd4bf;
      --accent-dark: #14b8a6;
      --warm: #fbbf24;
      --danger: #fecdd3;
      --focus: rgba(45, 212, 191, 0.18);
      --code: #0c100e;
      --code-text: #d7e0db;
      --shadow: rgba(0, 0, 0, 0.24);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0 56px;
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }}
    .heading {{ min-width: 0; }}
    .eyebrow {{
      margin: 0 0 4px;
      color: var(--warm);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    .theme-toggle {{
      min-height: 40px;
      flex: 0 0 auto;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}
    .theme-toggle:hover {{
      border-color: var(--accent);
      background: var(--panel-soft);
    }}
    .theme-toggle:focus-visible {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--focus);
    }}
    .status {{
      min-height: 24px;
      margin: 0 0 14px;
      color: var(--muted);
      font-weight: 700;
    }}
    .discovery-panel {{
      display: grid;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .alert-box {{
      display: none;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      color: var(--ink);
      font-weight: 700;
    }}
    .alert-box.visible {{
      display: block;
    }}
    .discovery-summary {{
      display: none;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .discovery-summary.visible {{
      display: grid;
    }}
    .summary-card {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .summary-card strong {{
      display: block;
      font-size: 1.25rem;
    }}
    .summary-card span {{
      color: var(--muted);
      font-size: 0.88rem;
      font-weight: 700;
    }}
    .conflict-list {{
      display: none;
      margin: 0;
      padding: 0;
      list-style: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .conflict-list.visible {{
      display: block;
    }}
    .conflict-list li {{
      padding: 10px 14px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }}
    .conflict-list li:first-child {{
      border-top: 0;
    }}
    .conflict-list strong {{
      display: block;
      margin-bottom: 2px;
    }}
    .conflict-list span {{
      color: var(--muted);
      font-size: 0.88rem;
      font-weight: 700;
    }}
    section {{
      padding: 0 0 22px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 22px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) minmax(180px, auto) minmax(220px, auto) auto auto auto auto;
      gap: 10px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.9rem;
      font-weight: 700;
    }}
    input,
    select {{
      width: 100%;
      min-height: 52px;
      padding: 0 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font: inherit;
      color: var(--ink);
      background: var(--panel);
      outline: none;
      color-scheme: light;
    }}
    select {{
      font-weight: 700;
    }}
    :root[data-theme="dark"] input,
    :root[data-theme="dark"] select {{
      color-scheme: dark;
    }}
    input:focus-visible,
    select:focus-visible {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--focus);
    }}
    form {{ margin: 0; }}
    .checkbox-label {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 52px;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      font-size: 0.92rem;
      font-weight: 700;
    }}
    .checkbox-label input {{
      width: 18px;
      min-height: auto;
      height: 18px;
      margin: 0;
      padding: 0;
      flex: 0 0 auto;
    }}
    button {{
      min-height: 52px;
      border: 0;
      border-radius: 8px;
      padding: 0 22px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button.secondary {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
    }}
    button.secondary:hover {{
      border-color: var(--accent);
      background: var(--panel-soft);
    }}
    button.danger {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--danger);
    }}
    button.danger:hover {{
      border-color: var(--danger);
      background: var(--panel-soft);
    }}
    button:disabled {{
      cursor: wait;
      opacity: 0.7;
    }}
    pre {{
      margin: 0;
      min-height: 420px;
      max-height: calc(100vh - 260px);
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--code);
      color: var(--code-text);
      border-radius: 8px;
      padding: 14px;
      font: 13px/1.45 Consolas, "Cascadia Mono", monospace;
    }}
    @media (max-width: 900px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .controls {{ grid-template-columns: 1fr; }}
      button {{ width: 100%; }}
    }}
    @media (max-width: 520px) {{
      main {{
        width: min(100% - 20px, 1120px);
        padding-top: 24px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="heading">
      <p class="eyebrow">Local index</p>
      <h1>Video Index Setup</h1>
    </div>
    <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Switch to dark theme" aria-pressed="false">
      <span class="theme-label">Dark</span>
    </button>
  </header>
  <section>
    <div id="status" class="status">Status: {status}</div>
    <div class="discovery-panel">
      <div id="discovery-alert" class="alert-box"></div>
      <div id="discovery-summary" class="discovery-summary">
        <div class="summary-card"><strong id="discovery-new">0</strong><span>New</span></div>
        <div class="summary-card"><strong id="discovery-exact">0</strong><span>Exact matches</span></div>
        <div class="summary-card"><strong id="discovery-conflicts">0</strong><span>Filename conflicts</span></div>
      </div>
      <ul id="discovery-conflict-list" class="conflict-list"></ul>
    </div>
    <div class="controls">
      <form id="index-form" method="post" action="/run/index">
        <label>Input videos
          <input name="input" value="{input_value}">
        </label>
      </form>
      <label>Embedding
        <select form="index-form" name="embedding_backend">
          {embedding_options}
        </select>
      </label>
      <label class="checkbox-label">
        <input form="index-form" type="checkbox" name="reindex_existing" value="1" {reindex_checked}>
        Reindex exact matches
      </label>
      <form id="setup-form" method="post" action="/run/setup">
        <input type="hidden" name="input" value="{input_value}">
        <button class="secondary action-button" {disabled}>Setup models / binaries</button>
      </form>
      <button class="action-button" form="index-form" {disabled}>Index videos</button>
      <form id="web-form" method="post" action="/run/web">
        <input type="hidden" name="input" value="{input_value}">
        <input type="hidden" name="embedding_backend" value="{embedding_value}">
        <button class="secondary action-button" {disabled}>Run web search UI</button>
      </form>
      <form id="stop-form" method="post" action="/stop">
        <input type="hidden" name="input" value="{input_value}">
        <button id="stop-button" class="danger" {stop_disabled}>Stop</button>
      </form>
    </div>
  </section>
  <section>
    <pre id="runtime-log">{lines}</pre>
  </section>
</main>
<script>
  const themeToggle = document.querySelector("#theme-toggle");
  const themeLabel = themeToggle.querySelector(".theme-label");
  const statusEl = document.querySelector("#status");
  const logEl = document.querySelector("#runtime-log");
  const LOG_SCROLL_KEY = "bootstrap-runtime-log-scroll";
  const PAGE_SCROLL_KEY = "bootstrap-runtime-page-scroll";
  const discoveryAlert = document.querySelector("#discovery-alert");
  const discoverySummary = document.querySelector("#discovery-summary");
  const discoveryNew = document.querySelector("#discovery-new");
  const discoveryExact = document.querySelector("#discovery-exact");
  const discoveryConflicts = document.querySelector("#discovery-conflicts");
  const discoveryConflictList = document.querySelector("#discovery-conflict-list");
  const actionButtons = Array.from(document.querySelectorAll(".action-button"));
  const stopButton = document.querySelector("#stop-button");
  const embeddingSelect = document.querySelector("select[name='embedding_backend']");
  const reindexExisting = document.querySelector("#index-form input[name='reindex_existing']");
  const mirroredInputTargets = Array.from(document.querySelectorAll('form:not(#index-form) input[name="input"]'));
  const webEmbeddingInput = document.querySelector("#web-form input[name='embedding_backend']");
  const webInputPath = document.querySelector("#web-form input[name='input']");
  const inputPath = document.querySelector("#index-form input[name='input']");

  function preferredTheme() {{
    const saved = localStorage.getItem("video-search-theme");
    if (saved === "dark" || saved === "light") {{
      return saved;
    }}
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }}

  function applyTheme(theme) {{
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("video-search-theme", theme);
    const isDark = theme === "dark";
    themeToggle.setAttribute("aria-pressed", String(isDark));
    themeToggle.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");
    themeLabel.textContent = isDark ? "Light" : "Dark";
  }}

  themeToggle.addEventListener("click", () => {{
    const current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    applyTheme(current === "dark" ? "light" : "dark");
  }});

  applyTheme(preferredTheme());

  let lastLog = logEl.textContent;
  function currentLogScrollState() {{
    const nearBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 24;
    return nearBottom ? "bottom" : String(logEl.scrollTop);
  }}

  function saveLogScrollState() {{
    sessionStorage.setItem(LOG_SCROLL_KEY, currentLogScrollState());
  }}

  function restoreLogScrollState() {{
    const saved = sessionStorage.getItem(LOG_SCROLL_KEY);
    if (!saved) {{
      logEl.scrollTop = logEl.scrollHeight;
      return;
    }}
    if (saved === "bottom") {{
      logEl.scrollTop = logEl.scrollHeight;
      return;
    }}
    const scrollTop = Number.parseInt(saved, 10);
    if (Number.isFinite(scrollTop)) {{
      logEl.scrollTop = scrollTop;
    }}
  }}

  function savePageScrollState() {{
    sessionStorage.setItem(PAGE_SCROLL_KEY, String(window.scrollY || window.pageYOffset || 0));
  }}

  function restorePageScrollState() {{
    const saved = sessionStorage.getItem(PAGE_SCROLL_KEY);
    if (!saved) {{
      return;
    }}
    const scrollTop = Number.parseInt(saved, 10);
    if (Number.isFinite(scrollTop)) {{
      window.scrollTo(0, scrollTop);
    }}
  }}

  function renderDiscovery(discovery) {{
    const data = discovery || {{ new: 0, exact_matches: 0, conflicts: 0, items: [] }};
    const hasAny = (data.new || data.exact_matches || data.conflicts);
    discoverySummary.classList.toggle("visible", Boolean(hasAny));
    discoveryNew.textContent = String(data.new || 0);
    discoveryExact.textContent = String(data.exact_matches || 0);
    discoveryConflicts.textContent = String(data.conflicts || 0);

    const items = Array.isArray(data.items) ? data.items : [];
    if (items.length) {{
      discoveryAlert.classList.add("visible");
      discoveryAlert.textContent = `${{items.length}} filename conflict${{items.length === 1 ? "" : "s"}} detected. Existing output and DB rows are not being overwritten.`;
      discoveryConflictList.classList.add("visible");
      discoveryConflictList.innerHTML = items.map((item) => `
        <li>
          <strong>${{item.filename || "Unnamed file"}}</strong>
          <span>${{item.action || "processed separately"}}</span>
        </li>
      `).join("");
    }} else {{
      discoveryAlert.classList.remove("visible");
      discoveryAlert.textContent = "";
      discoveryConflictList.classList.remove("visible");
      discoveryConflictList.innerHTML = "";
    }}
  }}

  async function refreshStatus() {{
    try {{
      const response = await fetch("/api/status", {{ cache: "no-store" }});
      if (!response.ok) {{
        return;
      }}
      const payload = await response.json();
      statusEl.textContent = `Status: ${{payload.status}}`;
      renderDiscovery(payload.discovery);
      if (payload.log !== lastLog) {{
        const shouldStickToBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 24;
        logEl.textContent = payload.log;
        lastLog = payload.log;
        if (shouldStickToBottom) {{
          logEl.scrollTop = logEl.scrollHeight;
        }}
      }}
      for (const button of actionButtons) {{
        button.disabled = Boolean(payload.busy);
      }}
      stopButton.disabled = !payload.busy;
      if (payload.embedding_backend && embeddingSelect.value !== payload.embedding_backend) {{
        embeddingSelect.value = payload.embedding_backend;
        webEmbeddingInput.value = payload.embedding_backend;
      }}
      if (typeof payload.reindex_existing === "boolean") {{
        reindexExisting.checked = payload.reindex_existing;
      }}
    }} catch (error) {{
      statusEl.textContent = "Status: disconnected";
    }}
  }}

  embeddingSelect.addEventListener("change", () => {{
    webEmbeddingInput.value = embeddingSelect.value;
  }});

  inputPath.addEventListener("input", () => {{
    webInputPath.value = inputPath.value;
    for (const target of mirroredInputTargets) {{
      target.value = inputPath.value;
    }}
  }});

  logEl.addEventListener("scroll", saveLogScrollState);
  window.addEventListener("scroll", savePageScrollState);
  for (const form of document.querySelectorAll("form")) {{
    form.addEventListener("submit", () => {{
      saveLogScrollState();
      savePageScrollState();
    }});
  }}

  window.setInterval(refreshStatus, 500);
  restorePageScrollState();
  restoreLogScrollState();
  refreshStatus();
</script>
</body>
</html>"""


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        config.update(loaded)
        commands = dict(DEFAULT_CONFIG["commands"])
        commands.update(loaded.get("commands", {}))
        config["commands"] = commands
    return config


def python_executable() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    scripts_dir = ROOT / ".venv" / "Scripts"
    env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["BOOTSTRAP_RUNTIME_EVENTS"] = "1"
    return env


def open_search_ui_after_delay() -> None:
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")


def main() -> int:
    config = load_config()
    state = RuntimeState(config)
    state.log(f"Runtime menu: http://{MENU_HOST}:{MENU_PORT}")
    handler_type = type("ConfiguredRuntimeHandler", (RuntimeHandler,), {"state": state})
    server = ThreadingHTTPServer((MENU_HOST, MENU_PORT), handler_type)
    webbrowser.open(f"http://{MENU_HOST}:{MENU_PORT}")
    try:
        server.serve_forever()
    finally:
        state.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
