"""Backend boundary between MCP semantics and Polypack implementations."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

CLASSES = {"episodic", "semantic", "procedural", "entity"}
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

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "class": self.memory_class,
                "context": self.context, "confidence": self.confidence,
                "provenance": self.provenance, "activation": round(self.activation, 6),
                "createdAt": self.created_at, "metadata": self.metadata,
                "supersededBy": self.superseded_by, "suppressed": self.suppressed,
                "tokenEstimate": estimate_tokens(self.content)}


class MemoryBackend(Protocol):
    def store(self, content: str, **kwargs: Any) -> dict[str, Any]: ...
    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]: ...
    def context(self, context: str, **kwargs: Any) -> dict[str, Any]: ...
    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]: ...
    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def supersede(self, new_id: str, old_id: str) -> dict[str, Any]: ...
    def consolidate(self, source_ids: list[str], content: str, **kwargs: Any) -> dict[str, Any]: ...
    def graph_query(self, operation: str, **kwargs: Any) -> dict[str, Any]: ...
    def stats(self) -> dict[str, Any]: ...


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
        ranked, diagnostics = self._rank(query, context, kwargs.get("strict_context", False))
        return self._response(ranked[:limit], diagnostics, matchedContext=sum(1 for x in ranked if x["context"] == context) if context else 0,
                              fallbackAttempted=bool(context and any(x["context"] is None for x in ranked)))

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
        if operation == "schema": return {"nodes": ["memory"], "edges": sorted({e["type"] for e in self.edges})}
        raise ValueError(f"Unsupported graph operation: {operation}")

    def stats(self) -> dict[str, Any]:
        return {"memories": len(self.memories), "edges": len(self.edges)}


class PolypackBackend(InMemoryBackend):
    """Adapter using the real Python Polypack graph and activation engine."""

    def __init__(self, graph: Any | None = None) -> None:
        try:
            from polypack import ActivationEngine, PolyGraph
        except ImportError as exc:
            raise RuntimeError("Install polypack-db or use InMemoryBackend") from exc
        self.graph = graph or PolyGraph()
        self.engine = ActivationEngine(self.graph)

    def store(self, content: str, **kwargs: Any) -> dict[str, Any]:
        now = int(time.time() * 1000); memory_id = kwargs.get("id", str(uuid.uuid4()))
        node = {"id": memory_id, "type": "memory", "memoryClass": kwargs.get("memory_class", "semantic"),
                "data": {"content": content, "context": kwargs.get("context"), "provenance": kwargs.get("provenance", {}),
                         "confidence": kwargs.get("confidence", 1.0), "metadata": kwargs.get("metadata", {})},
                "insertedAt": now, "updatedAt": now}
        self.graph.add_node(node); self.graph.reinforce_node(memory_id, kwargs.get("activation", 0.1), "memory_store",
                                                             context=kwargs.get("context"))
        return self._node(memory_id)

    def _node(self, memory_id: str) -> dict[str, Any]:
        node = self.graph._nodes[memory_id]
        content = node.get("data", {}).get("content", "")
        return {"id": memory_id, "content": content, "class": node.get("memoryClass", "semantic"),
                "context": node.get("data", {}).get("context"), "confidence": node.get("data", {}).get("confidence", 1.0),
                "provenance": node.get("data", {}).get("provenance", {}), "activation": self.graph.get_activation(memory_id) or 0,
                "suppressed": bool(node.get("data", {}).get("suppressed", False)),
                "tokenEstimate": estimate_tokens(content)}

    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]:
        before = self.graph.get_activation(memory_id) or 0
        self.engine.record_feedback(memory_id, useful)
        self.graph.reinforce_node(memory_id, 0.1 if useful else -0.1, "mcp_feedback")
        after = self.graph.get_activation(memory_id) or 0
        return {"memory_id": memory_id, "useful": useful, "activation_before": before,
                "activation_after": after, "weights_changed": False}

    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        self.graph.suppress_node(memory_id, kwargs.get("amount", 0.5), "mcp_suppress")
        self.graph._nodes[memory_id].setdefault("data", {})["suppressed"] = True
        return {"id": memory_id, "activation": self.graph.get_activation(memory_id)}

    def supersede(self, new_id: str, old_id: str) -> dict[str, Any]:
        self.graph.supersede(new_id, old_id)
        return {"superseded": old_id, "by": new_id}

    def consolidate(self, source_ids: list[str], content: str, **kwargs: Any) -> dict[str, Any]:
        memory_id = kwargs.get("id", str(uuid.uuid4()))
        now = int(time.time() * 1000)
        node = {"id": memory_id, "type": "memory", "memoryClass": kwargs.get("memory_class", "semantic"),
                "data": {"content": content, "context": kwargs.get("context"),
                         "confidence": kwargs.get("confidence", 1.0), "derivedFrom": source_ids},
                "insertedAt": now, "updatedAt": now}
        self.graph.consolidate(node, source_ids)
        return self._node(memory_id)

    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]:
        # Embeddings are intentionally supplied by a future caller; lexical graph
        # traversal remains a useful fallback for text-only MCP clients.
        terms = set(query.lower().split())
        context, limit = kwargs.get("context"), kwargs.get("limit", 10)
        ranked = []
        for node_id, node in self.graph._nodes.items():
            item = self._node(node_id)
            if item.get("suppressed") or (context and (item["context"] != context if kwargs.get("strict_context") else item["context"] not in (None, context))): continue
            score = len(terms & set(item["content"].lower().split())) / max(len(terms), 1)
            score += item["activation"] * 0.25
            if score: ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        items = [{**item, "score": round(score, 6), "scoreComponents": {"lexical": round(score, 6)}} for score, item in ranked[:limit]]
        return {"items": items, "metadata": {"retrievalVersion": RETRIEVAL_VERSION,
                "candidateCount": len(ranked), "matchedContext": sum(1 for _, x in ranked if x["context"] == context) if context else 0,
                "excludedCount": max(0, len(self.graph._nodes) - len(ranked)), "fallbackAttempted": bool(context and any(x["context"] is None for _, x in ranked))}}

    def context(self, context: str, **kwargs: Any) -> dict[str, Any]:
        budget = kwargs.get("token_budget")
        if budget is not None and budget <= 0: raise ValueError("token_budget must be greater than zero")
        try:
            selected = self.engine.workingMemory({"limit": kwargs.get("limit", 10), "context": context,
                "tokenBudget": budget, "costOf": lambda memory: estimate_tokens(memory.content)})
            items = [self._node(item["id"] if isinstance(item, dict) else item.id) for item in selected]
        except (AttributeError, TypeError):
            ranked = self.recall("", context=context, strict_context=kwargs.get("strict_context", False), limit=100)["items"]
            used, items = 0, []
            for item in ranked:
                if budget is not None and used + item["tokenEstimate"] > budget: continue
                items.append(item); used += item["tokenEstimate"]
                if len(items) >= kwargs.get("limit", 10): break
        used = sum(item["tokenEstimate"] for item in items)
        return {"items": items, "metadata": {"retrievalVersion": RETRIEVAL_VERSION,
                "candidateCount": self.graph.size, "matchedContext": sum(1 for item in items if item["context"] == context),
                "budgetRequested": budget, "budgetUsed": used,
                "remainingBudget": None if budget is None else budget - used,
                "selectionCount": len(items), "fallbackAttempted": any(item["context"] is None for item in items)}}

    def graph_query(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        if operation == "schema": return {"nodes": ["memory"], "edges": sorted({e.get("type") for e in self.graph._edges.values() for e in e.values()})}
        if operation == "add_edge":
            self.graph.add_edge(kwargs["source"], kwargs["type"], kwargs["target"])
            return {"source": kwargs["source"], "type": kwargs["type"], "target": kwargs["target"]}
        if operation == "neighbors":
            node_id = kwargs["id"]
            edges = [edge for grouped in self.graph._edges.values() for edge in grouped.values()
                     if edge.get("source") == node_id or edge.get("target") == node_id]
            return {"id": node_id, "neighbors": edges}
        raise ValueError(f"Unsupported graph operation: {operation}")

    def stats(self) -> dict[str, Any]:
        return {"memories": self.graph.size, "edges": sum(len(edges) for edges in self.graph._edges.values())}
