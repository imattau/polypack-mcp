import pytest

pytest.importorskip("polypack")

from polypack import PolyGraph
from polypack_mcp.backend import PolypackBackend


def test_native_activation_working_memory_contract():
    backend = PolypackBackend(PolyGraph())
    global_memory = backend.store("global deployment guidance")
    project_memory = backend.store("project deployment guidance", context="project")

    selected = backend.context("project", token_budget=16)
    assert selected["metadata"]["activationLayer"] == "ActivationEngine.working_memory"
    assert selected["metadata"]["budgetUsed"] <= 16
    assert all(item["tokenEstimate"] <= 16 for item in selected["items"])
    assert {item["id"] for item in selected["items"]} == {global_memory["id"], project_memory["id"]}

    isolated = backend.context("other", token_budget=16, strict_context=True)
    assert isolated["items"] == []
    assert isolated["reason"] == "no_context_match"


def test_native_graph_and_feedback_contract():
    backend = PolypackBackend(PolyGraph())
    old = backend.store("old fact", context="project")
    new = backend.store("new fact", context="project")
    backend.supersede(new["id"], old["id"])
    summary = backend.consolidate([new["id"]], "summary fact", context="project")

    schema = backend.graph_query("schema")
    assert "SUPERSEDED_BY" in schema["edges"]
    assert "CONSOLIDATED_FROM" in schema["edges"]
    assert backend.graph_query("neighbors", id=summary["id"])["neighbors"]

    feedback = backend.feedback(new["id"], False)
    assert feedback["memory_id"] == new["id"]
    assert "activation_before" in feedback and "activation_after" in feedback


def test_durable_mutations_are_checkpointed_before_backend_returns(tmp_path):
    store_path = tmp_path / "store"
    backend = PolypackBackend(PolyGraph.open(store_path))
    memory = backend.store("durable memory", context="project")
    backend.close_store()

    reopened = PolypackBackend(PolyGraph.open(store_path))
    try:
        assert reopened.recall("durable memory", context="project")["items"][0]["id"] == memory["id"]
    finally:
        reopened.close_store()
