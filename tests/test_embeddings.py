import pytest

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


def test_provider_failure_falls_back_to_lexical_recall():
    class BrokenProvider(FakeProvider):
        def embed(self, texts):
            raise RuntimeError("helper offline")

    backend = PolypackBackend(PolyGraph(), embedding_provider=BrokenProvider())
    memory = backend.store("exact lexical recovery phrase")
    result = backend.recall("exact lexical recovery phrase")

    assert result["items"][0]["id"] == memory["id"]
    assert backend.stats()["embedding"]["status"] == "unavailable"


def test_context_updates_refresh_the_embedding():
    provider = FakeProvider()
    backend = PolypackBackend(PolyGraph(), embedding_provider=provider)
    memory = backend.store("A note", context="database")

    backend.update(memory["id"], {"context": "other"})

    assert backend.graph._nodes[memory["id"]]["vector"] == [0.0, 1.0]
