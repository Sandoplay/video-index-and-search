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


class RuntimeState:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_path = "data/input"
        self.embedding_backend = normalize_embedding_backend(
            str(config.get("default_embedding_backend") or "qwen3-4b-gguf")
        )
        self.lines: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

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

        replacements = {
            "embedding_backend": self.embedding_backend,
            **(replacements or {}),
        }
        resolved = [part.format(**replacements) for part in command]
        script = ROOT / resolved[0]
        args = [str(python_executable()), str(script), *resolved[1:]]
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
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if name == "web":
                threading.Thread(target=open_search_ui_after_delay, daemon=True).start()
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.log(line.rstrip())
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
        busy = self.state.is_busy()
        return {
            "busy": busy,
            "status": "Running" if busy else "Ready",
            "input_path": self.state.input_path,
            "embedding_backend": self.state.embedding_backend,
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
    section {{
      padding: 0 0 22px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 22px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) minmax(180px, auto) auto auto auto auto;
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
      <form method="post" action="/run/setup"><button class="secondary action-button" {disabled}>Setup models / binaries</button></form>
      <button class="action-button" form="index-form" {disabled}>Index videos</button>
      <form id="web-form" method="post" action="/run/web">
        <input type="hidden" name="input" value="{input_value}">
        <input type="hidden" name="embedding_backend" value="{embedding_value}">
        <button class="secondary action-button" {disabled}>Run web search UI</button>
      </form>
      <form method="post" action="/stop"><button id="stop-button" class="danger" {stop_disabled}>Stop</button></form>
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
  const actionButtons = Array.from(document.querySelectorAll(".action-button"));
  const stopButton = document.querySelector("#stop-button");
  const embeddingSelect = document.querySelector("select[name='embedding_backend']");
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
  async function refreshStatus() {{
    try {{
      const response = await fetch("/api/status", {{ cache: "no-store" }});
      if (!response.ok) {{
        return;
      }}
      const payload = await response.json();
      statusEl.textContent = `Status: ${{payload.status}}`;
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
    }} catch (error) {{
      statusEl.textContent = "Status: disconnected";
    }}
  }}

  embeddingSelect.addEventListener("change", () => {{
    webEmbeddingInput.value = embeddingSelect.value;
  }});

  inputPath.addEventListener("input", () => {{
    webInputPath.value = inputPath.value;
  }});

  window.setInterval(refreshStatus, 500);
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
