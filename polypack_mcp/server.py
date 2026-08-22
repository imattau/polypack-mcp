"""FastMCP server for Polypack adaptive memory."""

import argparse
import json
from contextlib import contextmanager
from threading import Condition, RLock


class ReadWriteLock:
    """Allow concurrent reads while keeping graph mutations exclusive."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self):
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()
from .backend import InMemoryBackend, MemoryBackend
from .service import MemoryService
from .setup import install_user_service, print_client_config

def create_server(backend: MemoryBackend | None = None, host: str = "127.0.0.1", port: int = 8765):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -e '.[polypack]'") from exc
    backend = backend or InMemoryBackend()
    service = MemoryService(backend)
    # FastMCP may execute synchronous tools in worker threads. Polypack's
    # graph transaction boundary is process-local: reads may overlap, but
    # mutations must exclude both reads and other mutations.
    operation_lock = ReadWriteLock()
    mcp = FastMCP("Polypack", host=host, port=port)

    @mcp.tool()
    def memory_store(content: str, memory_class: str = "semantic", context: str | None = None,
                     confidence: float = 1.0, provenance: dict | None = None, metadata: dict | None = None) -> dict:
        """Store durable adaptive memory with provenance and confidence."""
        with operation_lock.write():
            return service.store(content, memory_class, context, confidence, provenance, metadata)

    @mcp.tool()
    def memory_recall(query: str, context: str | None = None, limit: int = 10,
                      strict_context: bool = False) -> dict:
        """Hybrid semantic, graph, and activation-weighted retrieval."""
        with operation_lock.read():
            return backend.recall(query, context=context, strict_context=strict_context,
                                  limit=max(1, min(limit, 100)))

    @mcp.tool()
    def memory_context(context: str, limit: int = 10, token_budget: int | None = None,
                       strict_context: bool = False) -> dict:
        """Return a working-memory set selected by activation and estimated-token budget.

        token_budget is an estimated-token budget. Each returned memory fits wholly
        within the remaining budget; budgets must be greater than zero.
        """
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        with operation_lock.read():
            return backend.context(context, limit=max(1, min(limit, 100)), token_budget=token_budget,
                                   strict_context=strict_context)

    @mcp.tool()
    def memory_feedback(memory_id: str, useful: bool, agent_id: str = "default") -> dict:
        """Record whether a retrieved memory helped this agent session."""
        with operation_lock.write():
            return backend.feedback(memory_id, useful, agent_id=agent_id)

    @mcp.tool()
    def memory_suppress(memory_id: str, amount: float = 0.5) -> dict:
        """Inhibit a stale or unhelpful memory without deleting it."""
        with operation_lock.write():
            return backend.suppress(memory_id, amount=amount)

    @mcp.tool()
    def memory_supersede(new_memory_id: str, old_memory_id: str) -> dict:
        """Replace an outdated fact while retaining its history."""
        with operation_lock.write():
            return backend.supersede(new_memory_id, old_memory_id)

    @mcp.tool()
    def memory_consolidate(source_ids: list[str], content: str, context: str | None = None,
                           memory_class: str = "semantic", confidence: float = 1.0) -> dict:
        """Consolidate episodic memories into a durable higher-level memory."""
        with operation_lock.write():
            return backend.consolidate(source_ids, content, context=context, memory_class=memory_class, confidence=confidence)

    @mcp.tool()
    def graph_query(operation: str, id: str | None = None, source: str | None = None,
                    target: str | None = None, type: str | None = None) -> dict:
        """Escape hatch for narrowly scoped graph operations (neighbors, add_edge, schema)."""
        args = {k: v for k, v in {"id": id, "source": source, "target": target, "type": type}.items() if v is not None}
        lock = operation_lock.write if operation == "add_edge" else operation_lock.read
        with lock():
            return backend.graph_query(operation, **args)

    @mcp.resource("polypack://memory/context/{context}")
    def context_resource(context: str) -> str:
        with operation_lock.read():
            return json.dumps(backend.context(context), indent=2)

    @mcp.resource("polypack://memory/active")
    def active_resource() -> str:
        with operation_lock.read():
            return json.dumps(backend.context("", limit=20), indent=2)

    @mcp.resource("polypack://graph/schema")
    def schema_resource() -> str:
        with operation_lock.read():
            return json.dumps(backend.graph_query("schema"), indent=2)

    @mcp.resource("polypack://stats")
    def stats_resource() -> str:
        with operation_lock.read():
            return json.dumps(backend.stats(), indent=2)
    return mcp

def main() -> None:
    parser = argparse.ArgumentParser(description="Polypack adaptive-memory MCP server")
    parser.add_argument("command", nargs="?", choices=("serve", "setup"), default="serve")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--store", help="Polypack directory for durable storage")
    parser.add_argument("--host", default="127.0.0.1", help="Host used by the SSE server")
    parser.add_argument("--port", type=int, default=8765, help="Port used by the shared SSE endpoint")
    parser.add_argument("--no-start", action="store_true", help="Write the user service without starting it")
    args = parser.parse_args()
    if args.command == "setup":
        store = args.store or "~/.local/share/polypack-mcp"
        unit = install_user_service(store, port=args.port, start=not args.no_start)
        print(f"Installed {unit}")
        print_client_config(args.port)
        return
    backend = None
    if args.store:
        from .backend import PolypackBackend
        from polypack import PolyGraph
        backend = PolypackBackend(PolyGraph.open(args.store))
    create_server(backend, host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
