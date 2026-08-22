from polypack_mcp import InMemoryBackend, MemoryService

def test_store_recall_feedback_and_supersede():
    backend = InMemoryBackend(); service = MemoryService(backend)
    old = service.store("Python graph queries need optimisation", context="polypack")
    new = service.store("Python graph queries now use the native index", context="polypack")
    found = backend.recall("What performance work remains?", context="polypack")
    assert found and found[0]["id"] == old["id"]
    backend.feedback(old["id"], True)
    backend.supersede(new["id"], old["id"])
    assert all(item["id"] != old["id"] for item in backend.recall("graph queries", context="polypack"))

def test_context_budget_and_graph():
    backend = InMemoryBackend(); service = MemoryService(backend)
    a = service.store("one two", context="coding"); b = service.store("three four five", context="coding")
    assert len(backend.context("coding", token_budget=2)) == 1
    backend.graph_query("add_edge", source=a["id"], target=b["id"], type="related")
    assert len(backend.graph_query("neighbors", id=a["id"])["neighbors"]) == 1

