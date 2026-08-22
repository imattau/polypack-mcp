"""Backend boundary between MCP semantics and Polypack implementations."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

CLASSES = {"episodic", "semantic", "procedural", "entity"}


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

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "class": self.memory_class,
                "context": self.context, "confidence": self.confidence,
                "provenance": self.provenance, "activation": round(self.activation, 6),
                "createdAt": self.created_at, "metadata": self.metadata,
                "supersededBy": self.superseded_by}


class MemoryBackend(Protocol):
    def store(self, content: str, **kwargs: Any) -> dict[str, Any]: ...
    def recall(self, query: str, **kwargs: Any) -> list[dict[str, Any]]: ...
    def context(self, context: str, **kwargs: Any) -> list[dict[str, Any]]: ...
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

    def recall(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        terms = set(query.lower().split())
        context, limit = kwargs.get("context"), kwargs.get("limit", 10)
        ranked = []
        for memory in self.memories.values():
            if memory.superseded_by or (context and memory.context not in (None, context)):
                continue
            words = set(memory.content.lower().split())
            lexical = len(terms & words) / max(len(terms), 1)
            contextual = 0.15 if context and memory.context == context else 0
            score = lexical * 0.65 + memory.activation * 0.25 + memory.confidence * 0.1 + contextual
            if score > 0:
                ranked.append((score, memory))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [{**memory.as_dict(), "score": round(score, 6)} for score, memory in ranked[:limit]]

    def context(self, context: str, **kwargs: Any) -> list[dict[str, Any]]:
        items = [m for m in self.memories.values() if (not context or m.context == context) and not m.superseded_by]
        items.sort(key=lambda m: m.activation, reverse=True)
        budget, used = kwargs.get("token_budget"), 0
        result = []
        for memory in items:
            cost = len(memory.content.split())
            if budget is not None and used + cost > budget:
                continue
            used += cost
            result.append(memory.as_dict())
            if len(result) >= kwargs.get("limit", 10):
                break
        return result

    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]:
        memory = self._get(memory_id)
        memory.activation = min(1.0, max(0.0, memory.activation + (0.1 if useful else -0.1)))
        return {"id": memory_id, "useful": useful, "activation": memory.activation}

    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        memory = self._get(memory_id)
        memory.activation = max(0.0, memory.activation - kwargs.get("amount", 0.5))
        return {"id": memory_id, "activation": memory.activation}

    def supersede(self, new_id: str, old_id: str) -> dict[str, Any]:
        old = self._get(old_id); self._get(new_id); old.superseded_by = new_id; old.activation = 0.0
        return {"superseded": old_id, "by": new_id}

    def consolidate(self, source_ids: list[str], content: str, **kwargs: Any) -> dict[str, Any]:
        for source_id in source_ids: self._get(source_id)
        return self.store(content, memory_class=kwargs.get("memory_class", "semantic"),
                          context=kwargs.get("context"), confidence=kwargs.get("confidence", 1.0),
                          provenance={"derivedFrom": source_ids}, activation=0.7)

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
        return {"id": memory_id, "content": node.get("data", {}).get("content", ""), "class": node.get("memoryClass", "semantic"),
                "context": node.get("data", {}).get("context"), "confidence": node.get("data", {}).get("confidence", 1.0),
                "provenance": node.get("data", {}).get("provenance", {}), "activation": self.graph.get_activation(memory_id) or 0}

    def feedback(self, memory_id: str, useful: bool, **kwargs: Any) -> dict[str, Any]:
        self.engine.record_feedback(memory_id, useful)
        self.graph.reinforce_node(memory_id, 0.1 if useful else -0.1, "mcp_feedback")
        return {"id": memory_id, "useful": useful, "activation": self.graph.get_activation(memory_id)}

    def suppress(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        self.graph.suppress_node(memory_id, kwargs.get("amount", 0.5), "mcp_suppress")
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

    def recall(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        # Embeddings are intentionally supplied by a future caller; lexical graph
        # traversal remains a useful fallback for text-only MCP clients.
        terms = set(query.lower().split())
        context, limit = kwargs.get("context"), kwargs.get("limit", 10)
        ranked = []
        for node_id, node in self.graph._nodes.items():
            item = self._node(node_id)
            if context and item["context"] not in (None, context): continue
            score = len(terms & set(item["content"].lower().split())) / max(len(terms), 1)
            score += item["activation"] * 0.25
            if score: ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [{**item, "score": round(score, 6)} for score, item in ranked[:limit]]

    def context(self, context: str, **kwargs: Any) -> list[dict[str, Any]]:
        result = self.graph.top_activated(kwargs.get("limit", 10))
        items = [self._node(node["id"]) for node in result if not context or self._node(node["id"])["context"] == context]
        return items

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
