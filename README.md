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

Pass `--store` to open a durable Polypack directory. Without it, the server uses
the in-memory reference backend, which is convenient for smoke tests.

## Development

```sh
pip install -e '.[dev]'
pytest
```

The test suite includes an MCP client/server protocol smoke test covering tool
discovery, memory storage, recall, and resource reads.
