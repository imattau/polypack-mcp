# polypack-mcp

An MCP server that exposes Polypack as persistent adaptive memory. MCP-specific
tools live here; the database remains an independent dependency.

## Install and run

The simplest installation is from PyPI:

```sh
python3 -m pip install 'polypack-mcp[polypack]'
```

For one MCP client, use the default stdio server configuration. For Claude and
Codex sharing the same durable memory, install once and create a long-running
user service:

```sh
polypack-mcp setup --store ~/.local/share/polypack-mcp
```

This starts a Streamable HTTP server at `http://127.0.0.1:8765/mcp`, restarts it after a
failure, and prints client configuration snippets. The setup command uses
`systemd --user`; on systems without systemd, start the server directly:

```sh
polypack-mcp --transport streamable-http --port 8765 --store ~/.local/share/polypack-mcp
```

In shared Streamable HTTP mode, configure both clients with the URL. Do not configure them
with a `command` and `--store`, since that starts two processes competing for
the same durable store.

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.polypack]
  url = "http://127.0.0.1:8765/mcp"
```

Claude Desktop:

```json
{
  "mcpServers": {
    "polypack": { "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

### Debian package

The Debian package installs and starts a system-level `polypack-mcp` service
automatically. It runs as the dedicated `polypack` user, stores data in
`/var/lib/polypack-mcp`, and exposes the same local Streamable HTTP endpoint:

```sh
sudo apt install ./polypack-mcp_<version>_amd64.deb
```

After installation, point Claude and Codex at
`http://127.0.0.1:8765/mcp`. The default port can be changed in
`/etc/default/polypack-mcp`, followed by a service restart. The service can be
managed with:

```sh
sudo systemctl status polypack-mcp
sudo systemctl restart polypack-mcp
```

The PyPI installation remains user-managed and uses `polypack-mcp setup` to
create a per-user service instead.

### RPM package

RPM-based distributions can install the matching `.rpm` asset from the
[GitHub release](https://github.com/imattau/polypack-mcp/releases):

```sh
sudo dnf install ./polypack-mcp-<version>-1.x86_64.rpm
```

The RPM package provides the same systemd service, store location, localhost
Streamable HTTP endpoint, and Python 3.12 requirement as the Debian package.

## Run manually

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
The `polypack` extra requires `polypack-db>=3.2.0` and uses its native
`ActivationEngine.working_memory` selector for context assembly.

## Development

```sh
pip install -e '.[dev]'
pytest
```

The test suite includes an MCP client/server protocol smoke test covering tool
discovery, memory storage, recall, and resource reads.

## Documentation

- [Getting started](docs/getting-started.md)
- [Operations and configuration](docs/operations.md)
- [Troubleshooting](docs/troubleshooting.md)
