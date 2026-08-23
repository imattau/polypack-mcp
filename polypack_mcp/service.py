"""Validated, agent-facing operations shared by MCP transports and tests."""

from __future__ import annotations

from typing import Any
from .backend import CLASSES, MemoryBackend

class MemoryService:
    def __init__(self, backend: MemoryBackend): self.backend = backend

    def store(self, content: str, memory_class: str = "semantic", context: str | None = None,
              confidence: float = 1.0, provenance: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if not content.strip(): raise ValueError("content must not be empty")
        if memory_class not in CLASSES:
            raise ValueError(f"memory_class must be one of: {', '.join(sorted(CLASSES))}")
        if not 0 <= confidence <= 1: raise ValueError("confidence must be between 0 and 1")
        return self.backend.store(content, memory_class=memory_class, context=context, confidence=confidence,
                                  provenance=provenance or {}, metadata=metadata or {})

    def store_batch(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for item in memories:
            content = item.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be a non-empty string")
            memory_class = item.get("memory_class", "semantic")
            if memory_class not in CLASSES:
                raise ValueError(f"memory_class must be one of: {', '.join(sorted(CLASSES))}")
            confidence = item.get("confidence", 1.0)
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")

            # Extract other fields
            context = item.get("context")
            provenance = item.get("provenance")
            metadata = item.get("metadata")

            # Store it
            res = self.backend.store(
                content,
                memory_class=memory_class,
                context=context,
                confidence=confidence,
                provenance=provenance or {},
                metadata=metadata or {},
                id=item.get("id")
            )
            results.append(res)
        return results
