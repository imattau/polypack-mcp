"""FastMCP server for Polypack adaptive memory."""

import argparse
import json
from .backend import InMemoryBackend, MemoryBackend
from .service import MemoryService

def create_server(backend: MemoryBackend | None = None):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -e '.[polypack]'") from exc
    backend = backend or InMemoryBackend()
    service = MemoryService(backend)
    mcp = FastMCP("Polypack")

    @mcp.tool()
    def memory_store(content: str, memory_class: str = "semantic", context: str | None = None,
                     confidence: float = 1.0, provenance: dict | None = None, metadata: dict | None = None) -> dict:
        """Store durable adaptive memory with provenance and confidence."""
        return service.store(content, memory_class, context, confidence, provenance, metadata)

    @mcp.tool()
    def memory_recall(query: str, context: str | None = None, limit: int = 10) -> list[dict]:
        """Hybrid semantic, graph, and activation-weighted retrieval."""
        return backend.recall(query, context=context, limit=max(1, min(limit, 100)))

    @mcp.tool()
    def memory_context(context: str, limit: int = 10, token_budget: int | None = None) -> list[dict]:
        """Return a budgeted working-memory set for a context."""
        return backend.context(context, limit=max(1, min(limit, 100)), token_budget=token_budget)

    @mcp.tool()
    def memory_feedback(memory_id: str, useful: bool, agent_id: str = "default") -> dict:
        """Record whether a retrieved memory helped this agent session."""
        return backend.feedback(memory_id, useful, agent_id=agent_id)

    @mcp.tool()
    def memory_suppress(memory_id: str, amount: float = 0.5) -> dict:
        """Inhibit a stale or unhelpful memory without deleting it."""
        return backend.suppress(memory_id, amount=amount)

    @mcp.tool()
    def memory_supersede(new_memory_id: str, old_memory_id: str) -> dict:
        """Replace an outdated fact while retaining its history."""
        return backend.supersede(new_memory_id, old_memory_id)

    @mcp.tool()
    def memory_consolidate(source_ids: list[str], content: str, context: str | None = None,
                           memory_class: str = "semantic", confidence: float = 1.0) -> dict:
        """Consolidate episodic memories into a durable higher-level memory."""
        return backend.consolidate(source_ids, content, context=context, memory_class=memory_class, confidence=confidence)

    @mcp.tool()
    def graph_query(operation: str, id: str | None = None, source: str | None = None,
                    target: str | None = None, type: str | None = None) -> dict:
        """Escape hatch for narrowly scoped graph operations (neighbors, add_edge, schema)."""
        args = {k: v for k, v in {"id": id, "source": source, "target": target, "type": type}.items() if v is not None}
        return backend.graph_query(operation, **args)

    @mcp.resource("polypack://memory/context/{context}")
    def context_resource(context: str) -> str: return json.dumps(backend.context(context), indent=2)

    @mcp.resource("polypack://memory/active")
    def active_resource() -> str: return json.dumps(backend.context("", limit=20), indent=2)

    @mcp.resource("polypack://graph/schema")
    def schema_resource() -> str: return json.dumps(backend.graph_query("schema"), indent=2)

    @mcp.resource("polypack://stats")
    def stats_resource() -> str: return json.dumps(backend.stats(), indent=2)
    return mcp

def main() -> None:
    parser = argparse.ArgumentParser(description="Polypack adaptive-memory MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--store", help="Polypack directory for durable storage")
    args = parser.parse_args()
    backend = None
    if args.store:
        from .backend import PolypackBackend
        from polypack import PolyGraph
        backend = PolypackBackend(PolyGraph.open(args.store))
    create_server(backend).run(transport=args.transport)


if __name__ == "__main__":
    main()
