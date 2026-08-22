"""Validated, agent-facing operations shared by MCP transports and tests."""

from __future__ import annotations

from typing import Any
from .backend import CLASSES, MemoryBackend

class MemoryService:
    def __init__(self, backend: MemoryBackend): self.backend = backend

    def store(self, content: str, memory_class: str = "semantic", context: str | None = None,
              confidence: float = 1.0, provenance: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if not content.strip(): raise ValueError("content must not be empty")
        if memory_class not in CLASSES: raise ValueError(f"class must be one of {sorted(CLASSES)}")
        if not 0 <= confidence <= 1: raise ValueError("confidence must be between 0 and 1")
        return self.backend.store(content, memory_class=memory_class, context=context, confidence=confidence,
                                  provenance=provenance or {}, metadata=metadata or {})

