from polypack_mcp import InMemoryBackend, MemoryService

def test_store_recall_feedback_and_supersede():
    backend = InMemoryBackend(); service = MemoryService(backend)
    old = service.store("Python graph queries need optimisation", context="polypack")
    new = service.store("Python graph queries now use the native index", context="polypack")
    found = backend.recall("What performance work remains?", context="polypack")
    assert found["items"]
    backend.feedback(old["id"], True)
    backend.supersede(new["id"], old["id"])
    assert all(item["id"] != old["id"] for item in backend.recall("graph queries", context="polypack")["items"])


def test_invalid_memory_class_error_names_parameter_and_allowed_values():
    service = MemoryService(InMemoryBackend())
    try:
        service.store("a preference", memory_class="preference")
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("invalid memory class should be rejected")
    assert message == "memory_class must be one of: entity, episodic, procedural, semantic"

def test_context_budget_and_graph():
    backend = InMemoryBackend(); service = MemoryService(backend)
    a = service.store("one two", context="coding"); b = service.store("three four five", context="coding")
    assert len(backend.context("coding", token_budget=2)["items"]) == 1
    backend.graph_query("add_edge", source=a["id"], target=b["id"], type="related")
    assert len(backend.graph_query("neighbors", id=a["id"])["neighbors"]) == 1


def test_context_soft_fallback_strict_isolation_and_budget_metadata():
    backend = InMemoryBackend(); service = MemoryService(backend)
    service.store("global guidance for deployments", context=None)
    service.store("a very long context-specific memory that cannot fit", context="missing")
    response = backend.context("missing", token_budget=20)
    assert response["metadata"]["fallbackAttempted"] is False  # exact context exists
    assert response["metadata"]["budgetUsed"] <= 20
    assert all(item["tokenEstimate"] <= 20 for item in response["items"])

    isolated = backend.context("other", strict_context=True, token_budget=20)
    assert isolated["items"] == []
    assert isolated["metadata"]["reason"] == "no_context_match"
    assert isolated["metadata"]["searchedContext"] == "other"


def test_context_fallback_and_feedback_suppression():
    backend = InMemoryBackend(); service = MemoryService(backend)
    global_memory = service.store("global release checklist")
    fallback = backend.context("release", token_budget=20)
    assert fallback["items"][0]["id"] == global_memory["id"]
    assert fallback["metadata"]["fallbackAttempted"] is True

    before = global_memory["activation"]
    feedback = backend.feedback(global_memory["id"], False)
    assert feedback["activation_before"] == before
    assert feedback["activation_after"] <= before
    backend.suppress(global_memory["id"])
    assert global_memory["id"] not in {item["id"] for item in backend.context("release")["items"]}


def test_recall_can_hydrate_bounded_relationship_neighbors():
    backend = InMemoryBackend(); service = MemoryService(backend)
    earlier = service.store("earlier handoff finding", context="cross-agent")
    latest = service.store("latest applied fix", context="cross-agent")
    backend.link(latest["id"], earlier["id"], "RESPONDS_TO")

    result = backend.recall(
        "latest applied fix", context="cross-agent", include_neighbors=True,
        edge_types=["RESPONDS_TO"], depth=1, limit=5, token_budget=1000,
    )

    assert [item["id"] for item in result["items"]] == [latest["id"], earlier["id"]]
    neighbor = result["items"][1]
    assert neighbor["retrievalRole"] == "neighbor"
    assert neighbor["distance"] == 1
    assert neighbor["relationship"]["edge"]["type"] == "RESPONDS_TO"
    assert result["metadata"]["neighborCount"] == 1


def test_recall_neighbor_limit_and_budget_are_enforced():
    backend = InMemoryBackend(); service = MemoryService(backend)
    first = service.store("primary finding", context="x")
    second = service.store("neighbor one", context="x")
    third = service.store("neighbor two", context="x")
    backend.link(first["id"], second["id"], "RELATED_TO")
    backend.link(first["id"], third["id"], "RELATED_TO")

    result = backend.recall("primary finding", context="x", include_neighbors=True,
                            edge_types=["RELATED_TO"], limit=2, token_budget=10)
    assert len(result["items"]) <= 2
    assert result["metadata"]["budgetUsed"] <= 10


def test_recall_reserves_capacity_for_neighbors_when_primary_matches_fill_limit():
    backend = InMemoryBackend(); service = MemoryService(backend)
    linked = service.store("process lesson RESPONDS_TO edge convention", context="cross-agent")
    neighbor = service.store("linked response memory", context="cross-agent")
    service.store("another process lesson RESPONDS_TO detail", context="cross-agent")
    backend.link(linked["id"], neighbor["id"], "RESPONDS_TO")

    result = backend.recall(
        "process lesson RESPONDS_TO edge convention", context="cross-agent",
        include_neighbors=True, edge_types=["RESPONDS_TO"], depth=1,
        limit=3, token_budget=2000,
    )

    assert len(result["items"]) == 3
    assert result["metadata"]["neighborCount"] == 1
    assert any(item["retrievalRole"] == "neighbor" for item in result["items"])


def test_recall_neighbor_limit_and_availability_metadata():
    backend = InMemoryBackend(); service = MemoryService(backend)
    primary = service.store("primary handoff", context="cross-agent")
    first = service.store("first linked response", context="cross-agent")
    second = service.store("second linked response", context="cross-agent")
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


def test_relationship_diagnostics_identifies_provenance_only_links():
    backend = InMemoryBackend(); service = MemoryService(backend)
    target = service.store("earlier finding", context="cross-agent")
    source = service.store("later response", context="cross-agent",
                           provenance={"responds_to": target["id"]})

    diagnostics = backend.graph_query("relationship_diagnostics")
    assert diagnostics["provenanceOnly"] == [[source["id"], target["id"]]]
    assert diagnostics["consistent"] is False

    backend.link(source["id"], target["id"], "RESPONDS_TO")
    assert backend.graph_query("relationship_diagnostics")["consistent"] is True


def test_supersede_and_consolidation_edges_are_traversable():
    backend = InMemoryBackend(); service = MemoryService(backend)
    first = service.store("first fact", context="x")
    second = service.store("replacement fact", context="x")
    backend.supersede(second["id"], first["id"])
    combined = backend.consolidate([first["id"], second["id"]], "combined fact", context="x")
    schema = backend.graph_query("schema")
    assert {"SUPERSEDES", "SUPERSEDED_BY", "CONSOLIDATED_FROM"} <= set(schema["edges"])
    neighbors = backend.graph_query("neighbors", id=combined["id"])["neighbors"]
    assert len(neighbors) == 2


def test_memory_thread():
    backend = InMemoryBackend(); service = MemoryService(backend)
    m1 = service.store("First message", context="thread-test")
    m2 = service.store("Second message (reply to first)", context="thread-test")
    m3 = service.store("Third message (reply to second)", context="thread-test")

    backend.link(m2["id"], m1["id"], "RESPONDS_TO")
    backend.link(m3["id"], m2["id"], "RESPONDS_TO")

    # Test walking from middle (m2)
    thread = backend.memory_thread(m2["id"])
    assert len(thread["items"]) == 3
    # Check chronological order (m1, m2, m3)
    assert [item["id"] for item in thread["items"]] == [m1["id"], m2["id"], m3["id"]]

    # Test max_depth limits traversal
    thread_shallow = backend.memory_thread(m3["id"], max_depth=1)
    assert len(thread_shallow["items"]) == 2
    assert [item["id"] for item in thread_shallow["items"]] == [m2["id"], m3["id"]]


def test_batch_operations():
    backend = InMemoryBackend(); service = MemoryService(backend)
    # Batch store
    memories_to_store = [
        {"content": "Batch memory 1", "context": "batch", "memory_class": "semantic"},
        {"content": "Batch memory 2", "context": "batch", "memory_class": "episodic"},
        {"content": "Batch memory 3", "context": "batch", "memory_class": "procedural"}
    ]
    stored = service.store_batch(memories_to_store)
    assert len(stored) == 3
    assert stored[0]["content"] == "Batch memory 1"
    assert stored[1]["class"] == "episodic"
    assert stored[2]["class"] == "procedural"

    # Batch link
    links_to_create = [
        {"source_memory_id": stored[1]["id"], "target_memory_id": stored[0]["id"], "relationship": "RESPONDS_TO"},
        {"source_memory_id": stored[2]["id"], "target_memory_id": stored[1]["id"]}  # Defaults to RESPONDS_TO
    ]
    linked = backend.link_batch(links_to_create)
    assert len(linked) == 2
    assert linked[0]["source"] == stored[1]["id"]
    assert linked[0]["target"] == stored[0]["id"]
    assert linked[0]["type"] == "RESPONDS_TO"
    assert linked[1]["source"] == stored[2]["id"]
    assert linked[1]["target"] == stored[1]["id"]
    assert linked[1]["type"] == "RESPONDS_TO"
