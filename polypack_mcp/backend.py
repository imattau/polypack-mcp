"""Backend boundary between MCP semantics and Polypack implementations."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .embeddings import EmbeddingProvider

CLASSES = {"episodic", "semantic", "procedural", "entity"}
MemoryClass = Literal["entity", "episodic", "procedural", "semantic"]
RETRIEVAL_VERSION = "2026-08-22"


def estimate_tokens(content: str) -> int:
    """Conservative, deterministic token estimate used by MCP budgets."""
    return max(1, math.ceil(len(content) / 4))


@dataclass
class Memory:
    id: str
    content: str
    memory_class: str = "semantic"
    context: str | None = None
    confidence: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    superseded_by: str | None = None
    suppressed: bool = False
    revision: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "class": self.memory_class,
                "context": self.context, "confidence": self.confidence,
                "provenance": self.provenance, "activation": round(self.activation, 6),
                "createdAt": self.created_at, "metadata": self.metadata,
                "supersededBy": self.superseded_by, "suppressed": self.suppressed,
                "tokenEstimate": estimate_tokens(self.content), "revision": self.revision}


class MemoryBackend(Protocol):
    def store(self, content: str, **kwargs: Any) -> dict[str, Any]: ...
    def get(self, memory_id: str) -> dict[str, Any]: ...
    def update(self, memory_id: str, patch: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...
    def list_contexts(self) -> dict[str, Any]: ...
    def delete(self, memory_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]: ...
    def context(self, context: str, **kwargs: Any) -> dict[str, Any]: ...
    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]: ...
    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def supersede(self, new_id: str, old_id: str) -> dict[str, Any]: ...
    def consolidate(self, source_ids: list[str], content: str, **kwargs: Any) -> dict[str, Any]: ...
    def graph_query(self, operation: str, **kwargs: Any) -> dict[str, Any]: ...
    def link(self, source_id: str, target_id: str, relationship: str) -> dict[str, Any]: ...
    def unlink(self, source_id: str, target_id: str, relationship: str | None = None) -> dict[str, Any]: ...
    def stats(self) -> dict[str, Any]: ...
    def memory_thread(self, start_id: str, max_depth: int = 10) -> dict[str, Any]: ...
    def link_batch(self, links: list[dict[str, Any]]) -> list[dict[str, Any]]: ...



class InMemoryBackend:
    """Small reference backend; useful for tests and MCP smoke tests."""

    def __init__(self) -> None:
        self.memories: dict[str, Memory] = {}
        self.edges: list[dict[str, Any]] = []

    def _get(self, memory_id: str) -> Memory:
        if memory_id not in self.memories:
            raise ValueError(f"Unknown memory: {memory_id}")
        return self.memories[memory_id]

    def store(self, content: str, **kwargs: Any) -> dict[str, Any]:
        memory = Memory(id=kwargs.get("id", str(uuid.uuid4())), content=content,
                        memory_class=kwargs.get("memory_class", "semantic"),
                        context=kwargs.get("context"), confidence=kwargs.get("confidence", 1.0),
                        provenance=kwargs.get("provenance", {}), metadata=kwargs.get("metadata", {}),
                        activation=kwargs.get("activation", 0.1))
        self.memories[memory.id] = memory
        return memory.as_dict()

    def get(self, memory_id: str) -> dict[str, Any]:
        return self._get(memory_id).as_dict()

    def update(self, memory_id: str, patch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        memory = self._get(memory_id)
        expected_revision = kwargs.get("expected_revision")
        if expected_revision is not None and memory.revision != expected_revision:
            raise ValueError(f"memory {memory_id} has revision {memory.revision}, expected {expected_revision}")
        allowed = {"context", "confidence", "provenance", "metadata"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"update only supports: {', '.join(sorted(allowed))}")
        if "confidence" in patch and not 0 <= patch["confidence"] <= 1:
            raise ValueError("confidence must be between 0 and 1")
        for field_name in allowed & patch.keys():
            setattr(memory, field_name, patch[field_name])
        memory.revision += 1
        return memory.as_dict()

    def list_contexts(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        unscoped = 0
        for memory in self.memories.values():
            if memory.context is None:
                unscoped += 1
            else:
                counts[memory.context] = counts.get(memory.context, 0) + 1
        return {"contexts": sorted(counts), "counts": counts, "unscopedCount": unscoped}

    def delete(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        memory = self._get(memory_id)
        expected_revision = kwargs.get("expected_revision")
        if expected_revision is not None and memory.revision != expected_revision:
            raise ValueError(f"memory {memory_id} has revision {memory.revision}, expected {expected_revision}")
        removed_edges = self._edges_for(memory_id)
        del self.memories[memory_id]
        self.edges = [edge for edge in self.edges if edge not in removed_edges]
        return {"deleted": memory_id, "removedEdges": len(removed_edges)}

    def _rank(self, query: str = "", context: str | None = None,
              strict_context: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
        terms = set(query.lower().split())
        candidates, excluded = [], 0
        for memory in self.memories.values():
            if memory.superseded_by or memory.suppressed:
                excluded += 1; continue
            if context and (memory.context != context if strict_context else memory.context not in (None, context)):
                excluded += 1; continue
            words = set(memory.content.lower().split())
            lexical = len(terms & words) / max(len(terms), 1) if terms else 0.0
            context_match = bool(context and memory.context == context)
            context_score = 0.2 if context_match else (0.04 if context and memory.context is None else 0.0)
            activation_score = memory.activation * 0.25
            confidence_score = memory.confidence * 0.1
            score = lexical * 0.65 + activation_score + confidence_score + context_score
            if query and score <= 0: continue
            item = {**memory.as_dict(), "score": round(score, 6), "scoreComponents": {
                "lexical": round(lexical, 6), "context": round(context_score, 6),
                "activation": round(activation_score, 6), "confidence": round(confidence_score, 6)}}
            candidates.append(item)
        candidates.sort(key=lambda item: (item["score"], item["activation"], item["createdAt"]), reverse=True)
        return candidates, {"candidateCount": len(candidates), "excludedCount": excluded}

    def _response(self, items: list[dict[str, Any]], diagnostics: dict[str, Any], **meta: Any) -> dict[str, Any]:
        metadata = {"retrievalVersion": RETRIEVAL_VERSION, **diagnostics, **meta}
        aliases = {"candidate_count": "candidateCount", "matched_context": "matchedContext",
                   "fallback_attempted": "fallbackAttempted", "excluded_count": "excludedCount",
                   "budget_requested": "budgetRequested", "used_budget": "budgetUsed",
                   "remaining_budget": "remainingBudget", "selection_count": "selectionCount"}
        metadata.update({alias: metadata[source] for alias, source in aliases.items() if source in metadata})
        response = {"items": items, "metadata": metadata}
        if "reason" in metadata: response["reason"] = metadata["reason"]
        if "searchedContext" in metadata: response["searched_context"] = metadata["searchedContext"]
        if "fallbackAttempted" in metadata: response["fallback_attempted"] = metadata["fallbackAttempted"]
        return response

    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]:
        context, limit = kwargs.get("context"), kwargs.get("limit", 10)
        include_neighbors = bool(kwargs.get("include_neighbors"))
        neighbor_limit = kwargs.get("neighbor_limit", 1)
        if neighbor_limit is None:
            neighbor_limit = 1
        if neighbor_limit < 0:
            raise ValueError("neighbor_limit must be zero or greater")
        ranked, diagnostics = self._rank(query, context, kwargs.get("strict_context", False))
        budget = kwargs.get("token_budget")
        if budget is not None and budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        primary = []
        used = 0
        primary_candidates = ranked
        primary_limit = limit
        if include_neighbors:
            lexical_matches = [item for item in ranked if item["scoreComponents"]["lexical"] > 0]
            primary_candidates = lexical_matches or ranked[:1]
            # Keep room for at least one hydrated neighbor. If no neighbor is
            # found, the unused slot is filled with another primary below.
            primary_limit = limit if neighbor_limit == 0 else max(1, limit - neighbor_limit)
        for item in primary_candidates:
            if len(primary) >= primary_limit:
                break
            if budget is not None and used + item["tokenEstimate"] > budget:
                continue
            item["retrievalRole"] = "primary"
            primary.append(item)
            used += item["tokenEstimate"]

        neighbor_count = 0
        more_neighbors = False
        if include_neighbors:
            primary, used, neighbor_count, more_neighbors = self._append_neighbors(
                primary, context=context, strict_context=kwargs.get("strict_context", False),
                limit=limit, token_budget=budget, edge_types=kwargs.get("edge_types"),
                depth=kwargs.get("depth", 1), neighbor_limit=neighbor_limit, used_budget=used)
            if len(primary) < limit:
                selected_ids = {item["id"] for item in primary}
                for item in primary_candidates:
                    if len(primary) >= limit:
                        break
                    if item["id"] in selected_ids:
                        continue
                    if budget is not None and used + item["tokenEstimate"] > budget:
                        continue
                    item["retrievalRole"] = "primary"
                    primary.append(item)
                    selected_ids.add(item["id"])
                    used += item["tokenEstimate"]
        diagnostics.update({"matchedContext": sum(1 for x in ranked if x["context"] == context) if context else 0,
                            "fallbackAttempted": bool(context and any(x["context"] is None for x in ranked)),
                            "budgetRequested": budget, "budgetUsed": used,
                            "remainingBudget": None if budget is None else budget - used,
                            "neighborCount": neighbor_count,
                            "includeNeighbors": include_neighbors,
                            "neighborLimit": neighbor_limit if include_neighbors else 0,
                            "moreNeighborsAvailable": more_neighbors})
        return self._response(primary, diagnostics)

    def _edges_for(self, memory_id: str) -> list[dict[str, Any]]:
        return [edge for edge in self.edges
                if edge["source"] == memory_id or edge["target"] == memory_id]

    def _append_neighbors(self, primary: list[dict[str, Any]], *, context: str | None,
                          strict_context: bool, limit: int, token_budget: int | None,
                          edge_types: list[str] | None, depth: int, neighbor_limit: int,
                          used_budget: int) -> tuple[list[dict[str, Any]], int, int, bool]:
        if not primary or depth <= 0:
            return primary, used_budget, 0, False
        selected_ids = {item["id"] for item in primary}
        frontier = set(selected_ids)
        neighbor_count = 0
        more_neighbors = False
        for distance in range(1, min(depth, 3) + 1):
            next_frontier: set[str] = set()
            for node_id in sorted(frontier):
                for edge in self._edges_for(node_id):
                    if edge_types and edge["type"] not in set(edge_types):
                        continue
                    neighbor_id = edge["target"] if edge["source"] == node_id else edge["source"]
                    if neighbor_id in selected_ids:
                        continue
                    next_frontier.add(neighbor_id)
                    memory = self.memories.get(neighbor_id)
                    if memory is None or memory.suppressed or memory.superseded_by:
                        continue
                    if context and (memory.context != context if strict_context else memory.context not in (None, context)):
                        continue
                    item = memory.as_dict()
                    cost = item["tokenEstimate"]
                    if (neighbor_count >= neighbor_limit or len(primary) >= limit or
                            (token_budget is not None and used_budget + cost > token_budget)):
                        more_neighbors = True
                        continue
                    item.update(retrievalRole="neighbor", distance=distance,
                                relationship={"edge": edge,
                                              "direction": "outgoing" if edge["source"] == node_id else "incoming"})
                    primary.append(item)
                    selected_ids.add(neighbor_id)
                    used_budget += cost
                    neighbor_count += 1
            frontier = next_frontier - selected_ids
            if len(primary) >= limit:
                break
        return primary, used_budget, neighbor_count, more_neighbors

    def link(self, source_id: str, target_id: str, relationship: str) -> dict[str, Any]:
        return self.graph_query("add_edge", source=source_id, target=target_id, type=relationship)

    def unlink(self, source_id: str, target_id: str, relationship: str | None = None) -> dict[str, Any]:
        self._get(source_id); self._get(target_id)
        removed = [edge for edge in self.edges
                   if edge["source"] == source_id and edge["target"] == target_id
                   and (relationship is None or edge["type"] == relationship)]
        self.edges = [edge for edge in self.edges if edge not in removed]
        return {"source": source_id, "target": target_id, "relationship": relationship,
                "removed": len(removed), "edges": removed}

    def memory_thread(self, start_id: str, max_depth: int = 10) -> dict[str, Any]:
        if start_id not in self.memories:
            raise ValueError(f"Unknown memory: {start_id}")
        visited = {start_id}
        frontier = {start_id}
        for _ in range(max_depth):
            next_frontier = set()
            for node_id in frontier:
                for edge in self._edges_for(node_id):
                    if edge["type"] == "RESPONDS_TO":
                        neighbor_id = edge["target"] if edge["source"] == node_id else edge["source"]
                        if neighbor_id not in visited:
                            next_frontier.add(neighbor_id)
                            visited.add(neighbor_id)
            if not next_frontier:
                break
            frontier = next_frontier
        items = []
        for node_id in visited:
            memory = self.memories.get(node_id)
            if memory and not memory.suppressed and not memory.superseded_by:
                items.append(memory.as_dict())
        items.sort(key=lambda x: x["createdAt"])
        return {"items": items}

    def link_batch(self, links: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for link_info in links:
            source = link_info["source_memory_id"]
            target = link_info["target_memory_id"]
            relationship = link_info.get("relationship", "RESPONDS_TO")
            res = self.link(source, target, relationship)
            results.append(res)
        return results

    def context(self, context: str, **kwargs: Any) -> dict[str, Any]:
        limit, budget = kwargs.get("limit", 10), kwargs.get("token_budget")
        if budget is not None and budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        ranked, diagnostics = self._rank("", context, kwargs.get("strict_context", False))
        # Prefer one item from each memory class before filling remaining slots
        # with the highest-ranked repeats (activation layer's diversity stage).
        diverse, seen_classes = [], set()
        for item in ranked:
            if item["class"] not in seen_classes:
                diverse.append(item); seen_classes.add(item["class"])
        diverse.extend(item for item in ranked if item not in diverse)
        ranked = diverse
        used, selected = 0, []
        for item in ranked:
            cost = item["tokenEstimate"]
            if budget is not None and used + cost > budget: continue
            selected.append(item); used += cost
            if len(selected) >= limit: break
        matched = sum(1 for x in ranked if x["context"] == context)
        fallback = bool(context and matched == 0 and any(x["context"] is None for x in ranked))
        diagnostics.update({"matchedContext": matched, "fallbackAttempted": fallback,
                            "budgetRequested": budget, "budgetUsed": used,
                            "remainingBudget": None if budget is None else budget - used,
                            "selectionCount": len(selected)})
        if not selected and context and matched == 0:
            diagnostics.update(reason="no_context_match", searchedContext=context)
        return self._response(selected, diagnostics)

    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]:
        memory = self._get(memory_id)
        before = memory.activation
        memory.activation = min(1.0, max(0.0, memory.activation + (0.1 if useful else -0.1)))
        return {"memory_id": memory_id, "useful": useful, "activation_before": before,
                "activation_after": memory.activation, "weights_changed": False}

    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        memory = self._get(memory_id)
        memory.suppressed = True
        memory.activation = max(0.0, memory.activation - kwargs.get("amount", 0.5))
        return {"id": memory_id, "activation": memory.activation}

    def supersede(self, new_id: str, old_id: str) -> dict[str, Any]:
        old = self._get(old_id); self._get(new_id); old.superseded_by = new_id; old.activation = 0.0
        self.edges.extend([
            {"source": new_id, "type": "SUPERSEDES", "target": old_id},
            {"source": old_id, "type": "SUPERSEDED_BY", "target": new_id},
        ])
        return {"superseded": old_id, "by": new_id}

    def consolidate(self, source_ids: list[str], content: str, **kwargs: Any) -> dict[str, Any]:
        for source_id in source_ids: self._get(source_id)
        result = self.store(content, memory_class=kwargs.get("memory_class", "semantic"),
                          context=kwargs.get("context"), confidence=kwargs.get("confidence", 1.0),
                          provenance={"derivedFrom": source_ids}, activation=0.7)
        for source_id in source_ids:
            self.edges.append({"source": result["id"], "type": "CONSOLIDATED_FROM", "target": source_id})
        return result

    def graph_query(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        if operation == "add_edge":
            self._get(kwargs["source"]); self._get(kwargs["target"])
            edge = {"source": kwargs["source"], "type": kwargs["type"], "target": kwargs["target"]}
            self.edges.append(edge); return edge
        if operation == "neighbors":
            node_id = kwargs["id"]
            return {"id": node_id, "neighbors": [e for e in self.edges if e["source"] == node_id or e["target"] == node_id]}
        if operation == "relationship_diagnostics":
            return self._relationship_diagnostics()
        if operation == "schema": return {"nodes": ["memory"], "edges": sorted({e["type"] for e in self.edges})}
        raise ValueError(f"Unsupported graph operation: {operation}")

    def stats(self) -> dict[str, Any]:
        return {"memories": len(self.memories), "edges": len(self.edges)}

    def _relationship_diagnostics(self) -> dict[str, Any]:
        edge_pairs = {(edge["source"], edge["target"]) for edge in self.edges
                      if edge["type"] == "RESPONDS_TO"}
        provenance_pairs = {(memory.id, memory.provenance["responds_to"])
                            for memory in self.memories.values()
                            if isinstance(memory.provenance.get("responds_to"), str)}
        return {"relationship": "RESPONDS_TO",
                "provenanceOnly": [list(pair) for pair in sorted(provenance_pairs - edge_pairs)],
                "edgeOnly": [list(pair) for pair in sorted(edge_pairs - provenance_pairs)],
                "consistent": provenance_pairs == edge_pairs}


class PolypackBackend(InMemoryBackend):
    """Adapter using the real Python Polypack graph and activation engine."""

    def __init__(self, graph: Any | None = None, embedding_provider: EmbeddingProvider | None = None) -> None:
        try:
            from polypack import ActivationEngine, PolyGraph
        except ImportError as exc:
            raise RuntimeError("Install polypack-db or use InMemoryBackend") from exc
        self.graph = graph or PolyGraph()
        self.engine = ActivationEngine(self.graph)
        self.embedding_provider = embedding_provider
        self._embedding_error: str | None = None

    def embedding_status(self) -> dict[str, Any]:
        if self.embedding_provider is None:
            return {"enabled": False, "status": "disabled"}
        try:
            descriptor = self.embedding_provider.descriptor()
            status = "unavailable" if self._embedding_error else "ready"
            result = {"enabled": True, "status": status, **descriptor}
            if self._embedding_error:
                result["error"] = self._embedding_error
            return result
        except Exception as exc:
            return {"enabled": True, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _embedding_text(content: str, context: str | None = None) -> str:
        return f"context: {context}\ncontent: {content}" if context else content

    def _embed_one(self, content: str, context: str | None = None) -> list[float] | None:
        if self.embedding_provider is None:
            return None
        try:
            self._embedding_error = None
            return self.embedding_provider.embed([self._embedding_text(content, context)])[0]
        except Exception as exc:
            self._embedding_error = f"{type(exc).__name__}: {exc}"
            return None

    def _embed_query(self, query: str) -> list[float] | None:
        if self.embedding_provider is None:
            return None
        try:
            self._embedding_error = None
            return self.embedding_provider.embed([query])[0]
        except Exception as exc:
            self._embedding_error = f"{type(exc).__name__}: {exc}"
            return None

    def _checkpoint(self) -> None:
        """Persist a mutation when the graph has an attached durable store."""
        if getattr(self.graph, "_store", None) is not None:
            self.graph.checkpoint()

    def close_store(self) -> None:
        """Flush and close the attached Polypack store, if any."""
        self.graph.close_store()

    def store(self, content: str, **kwargs: Any) -> dict[str, Any]:
        now = int(time.time() * 1000); memory_id = kwargs.get("id", str(uuid.uuid4()))
        context = kwargs.get("context")
        node = {"id": memory_id, "type": "memory", "memoryClass": kwargs.get("memory_class", "semantic"),
                "data": {"content": content, "context": context, "provenance": kwargs.get("provenance", {}),
                         "confidence": kwargs.get("confidence", 1.0), "metadata": kwargs.get("metadata", {})},
                "insertedAt": now, "updatedAt": now}
        vector = self._embed_one(content, context)
        if vector is not None:
            node["vector"] = vector
        self.graph.add_node(node); self.graph.reinforce_node(memory_id, kwargs.get("activation", 0.1), "memory_store",
                                                             context=context)
        self._checkpoint()
        return self._node(memory_id)

    def _node(self, memory_id: str) -> dict[str, Any]:
        if memory_id not in self.graph._nodes:
            raise ValueError(f"Unknown memory: {memory_id}")
        node = self.graph._nodes[memory_id]
        data = node.get("data", {})
        content = data.get("content", "")
        provenance = dict(data.get("provenance", {}))
        if node.get("derivedFrom") is not None:
            provenance.setdefault("derivedFrom", node["derivedFrom"])
        if node.get("supersedes") is not None:
            provenance.setdefault("supersedes", node["supersedes"])
        return {"id": memory_id, "content": content, "class": node.get("memoryClass", "semantic"),
                "context": data.get("context"), "confidence": node.get("confidence", data.get("confidence", 1.0)),
                "provenance": provenance, "activation": self.graph.get_activation(memory_id) or 0,
                "metadata": dict(data.get("metadata", {})),
                "createdAt": node.get("insertedAt", 0) / 1000.0,
                "suppressed": bool(data.get("suppressed", False)) or bool(self.engine.inhibition_of(memory_id) > 0),
                "tokenEstimate": estimate_tokens(content), "revision": int(node.get("revision", 0))}

    def get(self, memory_id: str) -> dict[str, Any]:
        return self._node(memory_id)

    def update(self, memory_id: str, patch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._node(memory_id)
        allowed = {"context", "confidence", "provenance", "metadata"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"update only supports: {', '.join(sorted(allowed))}")
        if "confidence" in patch and not 0 <= patch["confidence"] <= 1:
            raise ValueError("confidence must be between 0 and 1")
        update_kwargs: dict[str, Any] = {"data": dict(patch),
                                         "expected_revision": kwargs.get("expected_revision")}
        if "context" in patch and self.embedding_provider is not None:
            current = self.graph._nodes[memory_id].get("data", {})
            vector = self._embed_one(current.get("content", ""), patch["context"])
            if vector is not None:
                update_kwargs["vector"] = vector
        result = self.graph.update_node(memory_id, **update_kwargs)
        if result is None:
            raise ValueError(f"Unknown memory: {memory_id}")
        self._checkpoint()
        return self._node(memory_id)

    def list_contexts(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        unscoped = 0
        for node in self.graph._nodes.values():
            context = node.get("data", {}).get("context")
            if context is None:
                unscoped += 1
            else:
                counts[context] = counts.get(context, 0) + 1
        return {"contexts": sorted(counts), "counts": counts, "unscopedCount": unscoped}

    def delete(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        self._node(memory_id)
        removed_edges = len(self._native_edges_for(memory_id))
        self.graph.remove_node(memory_id, expected_revision=kwargs.get("expected_revision"))
        self._checkpoint()
        return {"deleted": memory_id, "removedEdges": removed_edges}

    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]:
        before = self.graph.get_activation(memory_id) or 0
        weights_before = self.engine.get_weights()
        self.engine.record_feedback(memory_id, useful)
        self.graph.reinforce_node(memory_id, 0.1 if useful else -0.1, "mcp_feedback")
        self._checkpoint()
        after = self.graph.get_activation(memory_id) or 0
        weights_after = self.engine.get_weights()
        result = {"memory_id": memory_id, "useful": useful, "activation_before": before,
                  "activation_after": after, "weights_changed": weights_before != weights_after}
        if weights_before != weights_after:
            result.update(weights_before=weights_before, weights_after=weights_after)
        return result

    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        self.graph.suppress_node(memory_id, kwargs.get("amount", 0.5), "mcp_suppress")
        self.graph._nodes[memory_id].setdefault("data", {})["suppressed"] = True
        self._checkpoint()
        return {"id": memory_id, "activation": self.graph.get_activation(memory_id)}

    def supersede(self, new_id: str, old_id: str) -> dict[str, Any]:
        self.graph.supersede(new_id, old_id)
        self._checkpoint()
        return {"superseded": old_id, "by": new_id}

    def consolidate(self, source_ids: list[str], content: str, **kwargs: Any) -> dict[str, Any]:
        memory_id = kwargs.get("id", str(uuid.uuid4()))
        now = int(time.time() * 1000)
        node = {"id": memory_id, "type": "memory", "memoryClass": kwargs.get("memory_class", "semantic"),
                "data": {"content": content, "context": kwargs.get("context"),
                         "confidence": kwargs.get("confidence", 1.0), "derivedFrom": source_ids},
                "insertedAt": now, "updatedAt": now}
        vector = self._embed_one(content, kwargs.get("context"))
        if vector is not None:
            node["vector"] = vector
        self.graph.consolidate(node, source_ids)
        self._checkpoint()
        return self._node(memory_id)

    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]:
        terms = set(query.lower().split())
        context, limit = kwargs.get("context"), kwargs.get("limit", 10)
        include_neighbors = bool(kwargs.get("include_neighbors"))
        neighbor_limit = kwargs.get("neighbor_limit", 1)
        if neighbor_limit is None:
            neighbor_limit = 1
        if neighbor_limit < 0:
            raise ValueError("neighbor_limit must be zero or greater")
        query_vector = self._embed_query(query)
        semantic_scores: dict[str, float] = {}
        if query_vector is not None:
            try:
                similar = self.graph.query().where_type("memory").similar_to(
                    query_vector, top_k=max(limit * 5, 50)
                ).to_list()
                for node in similar:
                    vector = node.get("vector")
                    if vector:
                        dot = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
                        semantic_scores[node["id"]] = max(0.0, min(1.0, dot))
            except Exception as exc:
                self._embedding_error = f"{type(exc).__name__}: {exc}"
                query_vector = None
        ranked = []
        for node_id, node in self.graph._nodes.items():
            item = self._node(node_id)
            if item.get("suppressed") or (context and (item["context"] != context if kwargs.get("strict_context") else item["context"] not in (None, context))): continue
            lexical = len(terms & set(item["content"].lower().split())) / max(len(terms), 1)
            semantic = semantic_scores.get(node_id, 0.0)
            if query_vector is not None:
                components = {"semantic": round(0.55 * semantic, 6), "lexical": round(0.30 * lexical, 6),
                              "activation": round(0.15 * item["activation"], 6)}
            else:
                components = {"lexical": round(lexical, 6), "activation": round(item["activation"] * 0.25, 6)}
            score = sum(components.values())
            if score: ranked.append((score, item, lexical, components))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        budget = kwargs.get("token_budget")
        if budget is not None and budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        items = []
        used = 0
        primary_candidates = ranked
        primary_limit = limit
        if include_neighbors:
            meaningful_matches = [entry for entry in ranked
                                   if entry[2] > 0 or entry[3].get("semantic", 0) > 0]
            primary_candidates = meaningful_matches or ranked[:1]
            # Keep room for at least one hydrated neighbor. If no neighbor is
            # found, the unused slot is filled with another primary below.
            primary_limit = limit if neighbor_limit == 0 else max(1, limit - neighbor_limit)
        for score, item, lexical, components in primary_candidates:
            if len(items) >= primary_limit:
                break
            if budget is not None and used + item["tokenEstimate"] > budget:
                continue
            item = {**item, "score": round(score, 6), "scoreComponents": components,
                    "retrievalRole": "primary"}
            items.append(item)
            used += item["tokenEstimate"]
        neighbor_count = 0
        more_neighbors = False
        if include_neighbors:
            items, used, neighbor_count, more_neighbors = self._append_native_neighbors(
                items, context=context, strict_context=kwargs.get("strict_context", False),
                limit=limit, token_budget=budget, edge_types=kwargs.get("edge_types"),
                depth=kwargs.get("depth", 1), neighbor_limit=neighbor_limit, used_budget=used)
            if len(items) < limit:
                selected_ids = {item["id"] for item in items}
                for score, item, lexical, components in primary_candidates:
                    if len(items) >= limit:
                        break
                    if item["id"] in selected_ids:
                        continue
                    if budget is not None and used + item["tokenEstimate"] > budget:
                        continue
                    item = {**item, "score": round(score, 6), "scoreComponents": components,
                            "retrievalRole": "primary"}
                    items.append(item)
                    selected_ids.add(item["id"])
                    used += item["tokenEstimate"]
        return {"items": items, "metadata": {"retrievalVersion": RETRIEVAL_VERSION,
                "candidateCount": len(ranked), "matchedContext": sum(1 for _, x, _, _ in ranked if x["context"] == context) if context else 0,
                "excludedCount": max(0, len(self.graph._nodes) - len(ranked)),
                "fallbackAttempted": bool(context and any(x["context"] is None for _, x, _, _ in ranked)),
                "budgetRequested": budget, "budgetUsed": used,
                "remainingBudget": None if budget is None else budget - used,
                "neighborCount": neighbor_count,
                "includeNeighbors": include_neighbors,
                "neighborLimit": neighbor_limit if include_neighbors else 0,
                "moreNeighborsAvailable": more_neighbors}}

    def _native_edges_for(self, memory_id: str) -> list[dict[str, Any]]:
        return [edge for grouped in self.graph._edges.values() for edge in grouped.values()
                if edge.get("source") == memory_id or edge.get("target") == memory_id]

    def _append_native_neighbors(self, primary: list[dict[str, Any]], *, context: str | None,
                                 strict_context: bool, limit: int, token_budget: int | None,
                                 edge_types: list[str] | None, depth: int, neighbor_limit: int,
                                 used_budget: int) -> tuple[list[dict[str, Any]], int, int, bool]:
        if not primary or depth <= 0:
            return primary, used_budget, 0, False
        selected_ids = {item["id"] for item in primary}
        frontier = set(selected_ids)
        neighbor_count = 0
        more_neighbors = False
        allowed_types = set(edge_types) if edge_types else None
        for distance in range(1, min(depth, 3) + 1):
            next_frontier: set[str] = set()
            for node_id in sorted(frontier):
                for edge in self._native_edges_for(node_id):
                    if allowed_types and edge.get("type") not in allowed_types:
                        continue
                    neighbor_id = edge["target"] if edge["source"] == node_id else edge["source"]
                    if neighbor_id in selected_ids:
                        continue
                    next_frontier.add(neighbor_id)
                    item = self._node(neighbor_id)
                    if item["suppressed"] or (context and (item["context"] != context if strict_context else item["context"] not in (None, context))):
                        continue
                    cost = item["tokenEstimate"]
                    if (neighbor_count >= neighbor_limit or len(primary) >= limit or
                            (token_budget is not None and used_budget + cost > token_budget)):
                        more_neighbors = True
                        continue
                    item.update(retrievalRole="neighbor", distance=distance,
                                relationship={"edge": edge,
                                              "direction": "outgoing" if edge["source"] == node_id else "incoming"})
                    primary.append(item)
                    selected_ids.add(neighbor_id)
                    used_budget += cost
                    neighbor_count += 1
            frontier = next_frontier - selected_ids
            if len(primary) >= limit:
                break
        return primary, used_budget, neighbor_count, more_neighbors

    def context(self, context: str, **kwargs: Any) -> dict[str, Any]:
        budget = kwargs.get("token_budget")
        if budget is not None and budget <= 0: raise ValueError("token_budget must be greater than zero")
        selected = self.engine.working_memory(limit=kwargs.get("limit", 10), context=context,
            context_fallback=not kwargs.get("strict_context", False), token_budget=budget,
            cost_of=lambda memory: estimate_tokens(memory.get("data", {}).get("content", "")))
        items = [self._node(item["id"]) for item in selected
                 if not self._node(item["id"])["suppressed"]]
        used = sum(item["tokenEstimate"] for item in items)
        matched = sum(1 for item in items if item["context"] == context)
        fallback = any(item["context"] is None for item in items)
        metadata = {"retrievalVersion": RETRIEVAL_VERSION,
                "candidateCount": self.graph.size, "matchedContext": sum(1 for item in items if item["context"] == context),
                "budgetRequested": budget, "budgetUsed": used,
                "remainingBudget": None if budget is None else budget - used,
                "selectionCount": len(items), "fallbackAttempted": fallback,
                "activationLayer": "ActivationEngine.working_memory"}
        if not items and context and matched == 0:
            metadata.update(reason="no_context_match", searchedContext=context)
        return {"items": items, "metadata": metadata, **({"reason": metadata["reason"], "searched_context": context} if "reason" in metadata else {})}

    def graph_query(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        if operation == "schema": return {"nodes": ["memory"], "edges": sorted({e.get("type") for e in self.graph._edges.values() for e in e.values()})}
        if operation == "relationship_diagnostics":
            edge_pairs = {(edge["source"], edge["target"])
                          for grouped in self.graph._edges.values() for edge in grouped.values()
                          if edge.get("type") == "RESPONDS_TO"}
            provenance_pairs = set()
            for memory_id, node in self.graph._nodes.items():
                responds_to = node.get("data", {}).get("provenance", {}).get("responds_to")
                if isinstance(responds_to, str):
                    provenance_pairs.add((memory_id, responds_to))
            return {"relationship": "RESPONDS_TO",
                    "provenanceOnly": [list(pair) for pair in sorted(provenance_pairs - edge_pairs)],
                    "edgeOnly": [list(pair) for pair in sorted(edge_pairs - provenance_pairs)],
                    "consistent": provenance_pairs == edge_pairs}
        if operation == "add_edge":
            self.graph.add_edge(kwargs["source"], kwargs["type"], kwargs["target"])
            self._checkpoint()
            return {"source": kwargs["source"], "type": kwargs["type"], "target": kwargs["target"]}
        if operation == "neighbors":
            node_id = kwargs["id"]
            edges = [edge for grouped in self.graph._edges.values() for edge in grouped.values()
                     if edge.get("source") == node_id or edge.get("target") == node_id]
            return {"id": node_id, "neighbors": edges}
        raise ValueError(f"Unsupported graph operation: {operation}")

    def link(self, source_id: str, target_id: str, relationship: str) -> dict[str, Any]:
        return self.graph_query("add_edge", source=source_id, target=target_id, type=relationship)

    def unlink(self, source_id: str, target_id: str, relationship: str | None = None) -> dict[str, Any]:
        self._node(source_id); self._node(target_id)
        removed = [edge for edge in self._native_edges_for(source_id)
                   if edge.get("source") == source_id and edge.get("target") == target_id
                   and (relationship is None or edge.get("type") == relationship)]
        for edge in removed:
            self.graph.remove_edge(edge["id"])
        if removed:
            self._checkpoint()
        return {"source": source_id, "target": target_id, "relationship": relationship,
                "removed": len(removed), "edges": removed}

    def memory_thread(self, start_id: str, max_depth: int = 10) -> dict[str, Any]:
        if start_id not in self.graph._nodes:
            raise ValueError(f"Unknown memory: {start_id}")
        visited = {start_id}
        frontier = {start_id}
        for _ in range(max_depth):
            next_frontier = set()
            for node_id in frontier:
                for edge in self._native_edges_for(node_id):
                    if edge.get("type") == "RESPONDS_TO":
                        neighbor_id = edge["target"] if edge["source"] == node_id else edge["source"]
                        if neighbor_id not in visited:
                            next_frontier.add(neighbor_id)
                            visited.add(neighbor_id)
            if not next_frontier:
                break
            frontier = next_frontier
        items = []
        for node_id in visited:
            node = self.graph._nodes.get(node_id)
            if node:
                item = self._node(node_id)
                if not item.get("suppressed"):
                    items.append(item)
        items.sort(key=lambda x: self.graph._nodes[x["id"]].get("insertedAt", 0))
        return {"items": items}

    def stats(self) -> dict[str, Any]:
        return {"memories": self.graph.size, "edges": sum(len(edges) for edges in self.graph._edges.values()),
                "embedding": self.embedding_status(),
                "embeddingError": self._embedding_error}
