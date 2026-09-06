import pytest
from pathlib import Path

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
