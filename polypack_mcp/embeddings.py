"""Embedding providers and the optional local Qwen embedding helper."""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.request import Request, urlopen

DEFAULT_IDLE_TIMEOUT = 900.0


class EmbeddingProvider(Protocol):
    """Application-owned provider accepted by the Polypack adapter."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def descriptor(self) -> dict[str, Any]: ...


class HTTPEmbeddingProvider:
    """Provider client for a localhost embedding helper."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._descriptor: dict[str, Any] | None = None

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._request("/embed", {"texts": texts})
        vectors = result.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("embedding helper returned an invalid vector batch")
        return vectors

    def descriptor(self) -> dict[str, Any]:
        if self._descriptor is None:
            health = self.health()
            self._descriptor = {
                "provider": health.get("provider", "http"),
                "model": health.get("model", "unknown"),
                "dimension": health.get("dimension"),
                "version": health.get("version", "unknown"),
            }
        return dict(self._descriptor)


class _QwenHandler(BaseHTTPRequestHandler):
    model: Any = None
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    dimension = 1024
    load_error: str | None = None
    model_lock = threading.Lock()
    idle_timeout = DEFAULT_IDLE_TIMEOUT
    last_used: float = 0.0

    def log_message(self, format: str, *args: Any) -> None:
        print(f"embedding-helper: {format % args}", flush=True)

    @classmethod
    def load_model(cls) -> None:
        cls.last_used = time.monotonic()
        if cls.model is not None or cls.load_error is not None:
            return
        with cls.model_lock:
            if cls.model is not None or cls.load_error is not None:
                return
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                cls.model = SentenceTransformer(
                    cls.model_name, model_kwargs={"torch_dtype": torch.bfloat16}
                )
                cls.dimension = int(cls.model.get_sentence_embedding_dimension())
            except Exception as exc:  # startup health should explain dependency/model failures
                cls.load_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def unload_if_idle(cls) -> None:
        if cls.model is None:
            return
        if time.monotonic() - cls.last_used < cls.idle_timeout:
            return
        with cls.model_lock:
            if cls.model is None:
                return
            if time.monotonic() - cls.last_used < cls.idle_timeout:
                return
            cls.model = None
            gc.collect()
            print("embedding-helper: unloaded model after idle timeout", flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self.load_model()
        if self.load_error:
            self._json(503, {"status": "error", "error": self.load_error})
            return
        self._json(200, {
            "status": "ok",
            "provider": "qwen3",
            "model": self.model_name,
            "dimension": self.dimension,
            "version": "1",
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/embed":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            texts = payload.get("texts")
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError("texts must be a list of strings")
            self.load_model()
            if self.load_error:
                self._json(503, {"error": self.load_error})
                return
            vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=False)
            self._json(200, {"vectors": [[float(value) for value in vector] for vector in vectors]})
        except Exception as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def _idle_watchdog(stop: threading.Event, interval: float = 60.0) -> None:
    while not stop.wait(interval):
        _QwenHandler.unload_if_idle()


def helper_main(
    host: str = "127.0.0.1", port: int = 8766, idle_timeout: float = DEFAULT_IDLE_TIMEOUT
) -> None:
    _QwenHandler.idle_timeout = idle_timeout
    server = ThreadingHTTPServer((host, port), _QwenHandler)
    stop = threading.Event()
    watchdog = threading.Thread(target=_idle_watchdog, args=(stop,), daemon=True)
    watchdog.start()
    print(f"embedding-helper listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


def _helper_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Local Qwen embedding helper")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity before the model is unloaded (0 disables unloading)",
    )
    args = parser.parse_args(argv)
    idle_timeout = args.idle_timeout if args.idle_timeout > 0 else float("inf")
    helper_main(args.host, args.port, idle_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper_cli(os.sys.argv[1:]))
