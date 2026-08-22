# polypack-mcp

An MCP server that exposes Polypack as persistent adaptive memory. MCP-specific
tools live here; the database remains an independent dependency.

## Run

```sh
pip install -e '.[polypack]'
polypack-mcp --store ./polypack-data
```

The server exposes eight focused tools: `memory_store`, `memory_recall`,
`memory_context`, `memory_feedback`, `memory_suppress`, `memory_supersede`,
`memory_consolidate`, and `graph_query`. It also publishes context, active-memory,
schema, and stats resources under `polypack://`.

Retrieval tools return `{items, metadata}`. Metadata includes candidate and
excluded counts, context matches, score components, fallback behavior, the
retrieval version, and selection statistics. `memory_context` uses estimated
tokens (`ceil(content characters / 4)`, minimum one) as its `token_budget`.
An item is never returned if it would exceed the remaining budget; budgets less
than or equal to zero are rejected. Context is a soft preference: matching
memories are preferred and unscoped global memories may be used as fallback.
Pass `strict_context: true` for isolation. An empty isolated result reports
`reason: "no_context_match"` and the searched context.

Feedback is activation feedback: `useful=true` reinforces a memory and
`useful=false` provides negative retrieval feedback. Responses expose activation
before and after plus whether learned weights changed. Supersession and
consolidation materialize `SUPERSEDES`, `SUPERSEDED_BY`, and
`CONSOLIDATED_FROM` graph edges.

Pass `--store` to open a durable Polypack directory. Without it, the server uses
the in-memory reference backend, which is convenient for smoke tests.

## Development

```sh
pip install -e '.[dev]'
pytest
```

The test suite includes an MCP client/server protocol smoke test covering tool
discovery, memory storage, recall, and resource reads.
