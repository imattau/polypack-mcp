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


def test_native_recall_hydrates_responds_to_neighbor():
    backend = PolypackBackend(PolyGraph())
    earlier = backend.store("earlier handoff finding", context="cross-agent")
    latest = backend.store("latest applied fix", context="cross-agent")
    backend.link(latest["id"], earlier["id"], "RESPONDS_TO")

    result = backend.recall("latest applied fix", context="cross-agent",
                            include_neighbors=True, edge_types=["RESPONDS_TO"],
                            depth=1, limit=5, token_budget=1000)
    assert [item["id"] for item in result["items"]] == [latest["id"], earlier["id"]]
    assert result["items"][1]["retrievalRole"] == "neighbor"
    assert result["items"][1]["relationship"]["edge"]["type"] == "RESPONDS_TO"


def test_native_recall_reserves_capacity_for_neighbors_when_primary_matches_fill_limit():
    backend = PolypackBackend(PolyGraph())
    linked = backend.store("process lesson RESPONDS_TO edge convention", context="cross-agent")
    neighbor = backend.store("linked response memory", context="cross-agent")
    backend.store("another process lesson RESPONDS_TO detail", context="cross-agent")
    backend.link(linked["id"], neighbor["id"], "RESPONDS_TO")

    result = backend.recall(
        "process lesson RESPONDS_TO edge convention", context="cross-agent",
        include_neighbors=True, edge_types=["RESPONDS_TO"], depth=1,
        limit=3, token_budget=2000,
    )

    assert len(result["items"]) == 3
    assert result["metadata"]["neighborCount"] == 1
    assert any(item["retrievalRole"] == "neighbor" for item in result["items"])


def test_native_recall_neighbor_limit_reports_more_available():
    backend = PolypackBackend(PolyGraph())
    primary = backend.store("primary handoff", context="cross-agent")
    first = backend.store("first linked response", context="cross-agent")
    second = backend.store("second linked response", context="cross-agent")
    backend.link(primary["id"], first["id"], "RESPONDS_TO")
    backend.link(primary["id"], second["id"], "RESPONDS_TO")

    result = backend.recall(
        "primary handoff", context="cross-agent", include_neighbors=True,
        edge_types=["RESPONDS_TO"], depth=1, neighbor_limit=1,
        limit=4, token_budget=2000,
    )

    assert result["metadata"]["neighborCount"] == 1
    assert result["metadata"]["neighborLimit"] == 1
    assert result["metadata"]["moreNeighborsAvailable"] is True


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
