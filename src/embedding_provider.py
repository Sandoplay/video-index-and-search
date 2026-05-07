import os
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Protocol
from urllib.parse import urlparse

import requests


DB_PATH = Path("data/lancedb")
MODEL_CACHE_DIR = Path("models/.cache/huggingface")
MINILM_MODEL_NAME = "all-MiniLM-L6-v2"
MINILM_TABLE_NAME = "video_metadata"
MINILM_VECTOR_DIM = 384
DEFAULT_EMBEDDING_BACKEND = "qwen3-4b-gguf"

QWEN3_GGUF_MODEL_PATH = Path("models/Qwen3-Embedding-0.6B-Q8_0.gguf")
QWEN3_TABLE_NAME = "video_metadata_qwen3_1024"
QWEN3_VECTOR_DIM = 1024
QWEN3_4B_GGUF_MODEL_PATH = Path("models/Qwen3-Embedding-4B-Q8_0.gguf")
QWEN3_4B_TABLE_NAME = "video_metadata_qwen3_4b_2560"
QWEN3_4B_VECTOR_DIM = 2560
QWEN3_SERVER_URL = "http://127.0.0.1:8081"
QWEN3_CONTEXT_SIZE = 4096
QWEN3_QUERY_INSTRUCTION = (
    "Retrieve relevant local video summaries and timeline moments for the user's semantic search query."
)


class EmbeddingProvider(Protocol):
    name: str
    table_name: str
    vector_dim: int

    def embed_document(self, text: str) -> List[float]:
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

    def close(self) -> None:
        ...


class SentenceTransformerEmbeddingProvider:
    name = "minilm"
    table_name = MINILM_TABLE_NAME
    vector_dim = MINILM_VECTOR_DIM

    def __init__(
        self,
        model_name: str = MINILM_MODEL_NAME,
        cache_dir: str | Path = MODEL_CACHE_DIR,
    ):
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.cache_dir = str(cache_dir)
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)

    def embed_document(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_document(text)

    def close(self) -> None:
        return None


class LlamaCppEmbeddingProvider:
    name = "qwen3-gguf"
    table_name = QWEN3_TABLE_NAME
    vector_dim = QWEN3_VECTOR_DIM

    def __init__(
        self,
        model_path: str | Path = QWEN3_GGUF_MODEL_PATH,
        server_url: str = QWEN3_SERVER_URL,
        server_exe: str | Path = Path("bin/llama-cpp/llama-server.exe"),
        query_instruction: str = QWEN3_QUERY_INSTRUCTION,
        vector_dim: int = QWEN3_VECTOR_DIM,
        table_name: str = QWEN3_TABLE_NAME,
        context_size: int = QWEN3_CONTEXT_SIZE,
        parallel_slots: int = 1,
        autostart: bool = True,
        startup_timeout: int = 120,
    ):
        self.model_path = Path(model_path)
        self.server_url = server_url.rstrip("/")
        self.server_exe = Path(server_exe)
        self.query_instruction = query_instruction
        self.vector_dim = vector_dim
        self.table_name = table_name
        self.context_size = context_size
        self.parallel_slots = parallel_slots
        self.autostart = autostart
        self.startup_timeout = startup_timeout
        self.server_process: Optional[subprocess.Popen[Any]] = None
        self.is_external_server = False
        self.api_url = f"{self.server_url}/v1/embeddings"
        self.model_name = self.model_path.name

        if not self.model_path.exists():
            raise FileNotFoundError(f"Embedding GGUF model not found: {self.model_path}")
        if not self.server_exe.exists():
            raise FileNotFoundError(f"llama-server.exe not found: {self.server_exe}")

        self._ensure_server()

    def _server_port(self) -> str:
        parsed = urlparse(self.server_url)
        return str(parsed.port or 8080)

    def _is_server_ready(self) -> bool:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("status") == "ok"
        except requests.RequestException:
            return False

    def _ensure_server(self) -> None:
        if self._is_server_ready():
            self.is_external_server = True
            print(f">>> Connected to existing embedding server at {self.server_url}.")
            return

        if not self.autostart:
            raise RuntimeError(f"Embedding server is not running at {self.server_url}.")

        print(f">>> Starting embedding server: {self.model_path}")
        cmd = [
            str(self.server_exe.resolve()),
            "--model",
            str(self.model_path.resolve()),
            "--embedding",
            "--pooling",
            "last",
            "--host",
            "127.0.0.1",
            "--port",
            self._server_port(),
            "--n-gpu-layers",
            "-1",
            "--ubatch-size",
            str(min(self.context_size, 2048)),
            "--threads",
            "8",
            "--threads-batch",
            "8",
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            str(self.parallel_slots),
            "--no-webui",
        ]

        env = os.environ.copy()
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.server_process = subprocess.Popen(cmd, env=env, creationflags=creationflags)

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.server_process.poll() is not None:
                raise RuntimeError(f"Embedding server exited with code {self.server_process.returncode}")
            if self._is_server_ready():
                print(">>> Embedding server is ready.")
                return
            time.sleep(1)

        self.close()
        raise RuntimeError("Failed to start embedding server within timeout.")

    def _validate_vector(self, vector: List[Any]) -> List[float]:
        if len(vector) != self.vector_dim:
            raise RuntimeError(
                f"Expected {self.vector_dim}-dim embedding from {self.model_name}, got {len(vector)}. "
                "If another embedding server is already running on this port, stop it or set EMBEDDING_SERVER_URL."
            )
        return [float(value) for value in vector]

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload = {"model": self.model_name, "input": texts}
        response = requests.post(self.api_url, json=payload, timeout=120)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text.strip()
            if len(body) > 1000:
                body = body[:1000] + "..."
            raise RuntimeError(f"Embedding request failed: {body}") from exc

        data = response.json()
        items = data.get("data") or []
        if len(items) != len(texts):
            raise RuntimeError(f"Unexpected embedding response: {data}")

        ordered_items = sorted(
            enumerate(items),
            key=lambda pair: int(pair[1].get("index", pair[0])),
        )
        vectors: List[List[float]] = []
        for _, item in ordered_items:
            if "embedding" not in item:
                raise RuntimeError(f"Unexpected embedding response item: {item}")
            vectors.append(self._validate_vector(item["embedding"]))
        return vectors

    def _embed(self, text: str) -> List[float]:
        return self._embed_many([text])[0]

    def embed_document(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_many(texts)

    def embed_query(self, text: str) -> List[float]:
        instructed_query = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self._embed(instructed_query)

    def close(self) -> None:
        if self.server_process is None or self.is_external_server:
            return

        print(">>> Stopping embedding server...")
        self.server_process.terminate()
        try:
            self.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
        self.server_process = None


def create_embedding_provider(
    backend: str | None = None,
    model_name: str | None = None,
    cache_dir: str | Path = MODEL_CACHE_DIR,
    context_size: int | None = None,
    parallel_slots: int | None = None,
) -> EmbeddingProvider:
    selected_backend = (backend or os.getenv("EMBEDDING_BACKEND") or DEFAULT_EMBEDDING_BACKEND).strip().lower()

    if selected_backend in {"minilm", "sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbeddingProvider(
            model_name=model_name or os.getenv("EMBEDDING_MODEL_NAME") or MINILM_MODEL_NAME,
            cache_dir=cache_dir,
        )

    if selected_backend in {"qwen3", "qwen3-gguf", "qwen3-0.6b", "qwen3-0.6b-gguf", "llama-cpp", "llamacpp"}:
        model_path = os.getenv("EMBEDDING_GGUF_PATH") or str(QWEN3_GGUF_MODEL_PATH)
        server_url = os.getenv("EMBEDDING_SERVER_URL") or QWEN3_SERVER_URL
        query_instruction = os.getenv("EMBEDDING_QUERY_INSTRUCTION") or QWEN3_QUERY_INSTRUCTION
        vector_dim = int(os.getenv("EMBEDDING_VECTOR_DIM") or QWEN3_VECTOR_DIM)
        selected_context_size = context_size or int(os.getenv("EMBEDDING_CTX_SIZE") or QWEN3_CONTEXT_SIZE)
        selected_parallel_slots = parallel_slots or int(os.getenv("EMBEDDING_PARALLEL") or 1)
        return LlamaCppEmbeddingProvider(
            model_path=model_path,
            server_url=server_url,
            query_instruction=query_instruction,
            vector_dim=vector_dim,
            table_name=QWEN3_TABLE_NAME,
            context_size=selected_context_size,
            parallel_slots=selected_parallel_slots,
        )

    if selected_backend in {"qwen3-4b", "qwen3-4b-gguf"}:
        model_path = os.getenv("EMBEDDING_GGUF_PATH") or str(QWEN3_4B_GGUF_MODEL_PATH)
        server_url = os.getenv("EMBEDDING_SERVER_URL") or QWEN3_SERVER_URL
        query_instruction = os.getenv("EMBEDDING_QUERY_INSTRUCTION") or QWEN3_QUERY_INSTRUCTION
        vector_dim = int(os.getenv("EMBEDDING_VECTOR_DIM") or QWEN3_4B_VECTOR_DIM)
        selected_context_size = context_size or int(os.getenv("EMBEDDING_CTX_SIZE") or QWEN3_CONTEXT_SIZE)
        selected_parallel_slots = parallel_slots or int(os.getenv("EMBEDDING_PARALLEL") or 1)
        return LlamaCppEmbeddingProvider(
            model_path=model_path,
            server_url=server_url,
            query_instruction=query_instruction,
            vector_dim=vector_dim,
            table_name=QWEN3_4B_TABLE_NAME,
            context_size=selected_context_size,
            parallel_slots=selected_parallel_slots,
        )

    raise ValueError(
        "Unsupported EMBEDDING_BACKEND. Use 'minilm', 'qwen3-gguf', or 'qwen3-4b-gguf'."
    )


def default_table_name_for_backend(backend: str | None = None) -> str:
    selected_backend = (backend or os.getenv("EMBEDDING_BACKEND") or DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if selected_backend in {"minilm", "sentence-transformers", "sentence_transformers"}:
        return MINILM_TABLE_NAME
    if selected_backend in {"qwen3", "qwen3-gguf", "qwen3-0.6b", "qwen3-0.6b-gguf", "llama-cpp", "llamacpp"}:
        return QWEN3_TABLE_NAME
    if selected_backend in {"qwen3-4b", "qwen3-4b-gguf"}:
        return QWEN3_4B_TABLE_NAME
    raise ValueError(
        "Unsupported EMBEDDING_BACKEND. Use 'minilm', 'qwen3-gguf', or 'qwen3-4b-gguf'."
    )
