import io
import pytest
from pathlib import Path
from urllib.error import HTTPError

import polypack_mcp.embedding_cli as embedding_cli
import polypack_mcp.embeddings as embeddings

pytest.importorskip("polypack")

from polypack import PolyGraph
from polypack_mcp.backend import PolypackBackend


class FakeProvider:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        return [[1.0, 0.0] if "database" in text.lower() else [0.0, 1.0] for text in texts]

    def descriptor(self):
        return {"provider": "fake", "model": "test", "dimension": 2, "version": "1"}


def test_provider_vectors_are_stored_and_used_for_recall():
    provider = FakeProvider()
    backend = PolypackBackend(PolyGraph(), embedding_provider=provider)
    database = backend.store("Persistent database memory")
    unrelated = backend.store("A completely different note")

    assert backend.graph._nodes[database["id"]]["vector"] == [1.0, 0.0]
    result = backend.recall("database persistence", limit=2)

    assert result["items"][0]["id"] == database["id"]
    assert unrelated["id"] in {item["id"] for item in result["items"]}
    assert backend.stats()["embedding"]["model"] == "test"

    top = result["items"][0]
    assert "semantic" in top["scoreComponents"]
    assert top["score"] == pytest.approx(sum(top["scoreComponents"].values()))


def test_include_neighbors_keeps_zero_lexical_semantic_match_as_primary():
    class TaggedProvider:
        def __init__(self, vectors):
            self.vectors = vectors

        def embed(self, texts):
            return [self.vectors[text] for text in texts]

        def descriptor(self):
            return {"provider": "fake", "model": "test", "dimension": 2, "version": "1"}

    query = "warm beverage appliance repair"
    semantic_only = "espresso machine broken"
    lexical_only = "floor plan requires repair crew"
    provider = TaggedProvider({
        query: [1.0, 0.0],
        semantic_only: [1.0, 0.0],
        lexical_only: [0.0, 1.0],
    })
    backend = PolypackBackend(PolyGraph(), embedding_provider=provider)
    semantic_match = backend.store(semantic_only)
    backend.store(lexical_only)

    result = backend.recall(query, limit=1, include_neighbors=True, neighbor_limit=1)

    assert result["items"][0]["id"] == semantic_match["id"]


def test_provider_failure_falls_back_to_lexical_recall():
    class BrokenProvider(FakeProvider):
        def embed(self, texts):
            raise RuntimeError("helper offline")

    backend = PolypackBackend(PolyGraph(), embedding_provider=BrokenProvider())
    memory = backend.store("exact lexical recovery phrase")
    result = backend.recall("exact lexical recovery phrase")

    assert result["items"][0]["id"] == memory["id"]
    assert backend.stats()["embedding"]["status"] == "unavailable"

    top = result["items"][0]
    assert "semantic" not in top["scoreComponents"]
    assert top["score"] == pytest.approx(sum(top["scoreComponents"].values()))


def test_context_updates_refresh_the_embedding():
    provider = FakeProvider()
    backend = PolypackBackend(PolyGraph(), embedding_provider=provider)
    memory = backend.store("A note", context="database")

    backend.update(memory["id"], {"context": "other"})

    assert backend.graph._nodes[memory["id"]]["vector"] == [0.0, 1.0]


def test_unload_if_idle_clears_model_after_timeout(monkeypatch):
    handler = embeddings._QwenHandler
    handler.model = object()
    handler.idle_timeout = 10.0
    handler.last_used = 0.0

    monkeypatch.setattr(embeddings.time, "monotonic", lambda: 5.0)
    handler.unload_if_idle()
    assert handler.model is not None

    monkeypatch.setattr(embeddings.time, "monotonic", lambda: 20.0)
    handler.unload_if_idle()
    assert handler.model is None

    handler.idle_timeout = embeddings.DEFAULT_IDLE_TIMEOUT


def test_system_reindex_restores_service_store_ownership(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(embedding_cli, "_reindex", lambda store, provider: calls.append(("reindex", store)) or 3)
    monkeypatch.setattr(
        embedding_cli.subprocess,
        "run",
        lambda args, **kwargs: calls.append(("run", args, kwargs)) or None,
    )

    store = Path(tmp_path)
    assert embedding_cli._reindex_with_service_ownership(store, object(), system=True) == 3

    assert calls == [
        ("reindex", store),
        ("run", ["chown", "-R", "polypack:polypack", str(store)], {"check": True}),
    ]


def test_system_reindex_uses_yunohost_service_ownership(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(embedding_cli, "_reindex", lambda store, provider: calls.append(("reindex", store)) or 3)
    monkeypatch.setattr(
        embedding_cli.subprocess,
        "run",
        lambda args, **kwargs: calls.append(("run", args, kwargs)) or None,
    )

    store = Path(tmp_path)
    assert embedding_cli._reindex_with_service_ownership(
        store, object(), system=True, service_user="polypack_mcp", service_group="polypack_mcp"
    ) == 3

    assert calls == [
        ("reindex", store),
        ("run", ["chown", "-R", "polypack_mcp:polypack_mcp", str(store)], {"check": True}),
    ]


def test_helper_health_error_is_reported_without_waiting_for_timeout():
    error = HTTPError(
        "http://127.0.0.1:8766/health",
        503,
        "service unavailable",
        {},
        io.BytesIO(b'{"error":"model load failed"}'),
    )

    class BrokenProvider:
        def health(self):
            raise error

    with pytest.raises(RuntimeError, match="model load failed"):
        embedding_cli._wait_for_helper(BrokenProvider(), timeout=900)


def test_embedding_unit_and_main_dropin_use_allocated_port(monkeypatch, tmp_path):
    store = Path(tmp_path)
    venv = store / "venv"
    cache = store / "cache"
    unit_path = store / "systemd" / "polypack-mcp-embedding.service"
    dropin_path = store / "systemd" / "polypack-mcp.service.d" / "embedding.conf"
    monkeypatch.setattr(embedding_cli, "_unit_path", lambda system: unit_path)
    monkeypatch.setattr(embedding_cli, "_main_dropin", lambda system, service: dropin_path)
    original_unit = unit_path.read_text() if unit_path.exists() else None
    original_dropin = dropin_path.read_text() if dropin_path.exists() else None
    try:
        monkeypatch.setattr(
            embedding_cli.subprocess,
            "check_output",
            lambda *args, **kwargs: str(venv / "lib" / "python3.11" / "site-packages").encode(),
        )
        embedding_cli._write_unit(store, venv, cache, False, port=9876)
        embedding_cli._configure_main(False, True, "polypack-mcp.service", 9876)
        assert "--port 9876" in unit_path.read_text()
        assert "http://127.0.0.1:9876" in dropin_path.read_text()
    finally:
        if original_unit is None:
            unit_path.unlink(missing_ok=True)
        else:
            unit_path.write_text(original_unit)
        if original_dropin is None:
            dropin_path.unlink(missing_ok=True)
        else:
            dropin_path.parent.mkdir(parents=True, exist_ok=True)
            dropin_path.write_text(original_dropin)


def test_model_sources_accept_catalog_entries_and_hugging_face_repo_urls():
    assert embedding_cli._model_name("all-minilm-l6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_cli._model_name("custom", "https://huggingface.co/BAAI/bge-base-en-v1.5") == "BAAI/bge-base-en-v1.5"
    with pytest.raises(ValueError, match="single file"):
        embedding_cli._model_name("custom", "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors")


def test_storage_cleanup_prefers_suppressed_memories(tmp_path):
    store = tmp_path / "store"
    backend = PolypackBackend(PolyGraph.open(str(store)))
    suppressed = backend.store("suppressed memory")
    backend.store("ordinary memory")
    backend.suppress(suppressed["id"])
    before = backend.storage_usage_bytes()

    result = backend.cleanup_storage(max(1, before - 1))

    assert result["removed"] >= 1
    assert result["removedIds"][0] == suppressed["id"]
    backend.close_store()
