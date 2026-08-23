# Operations and configuration

## Service lifecycle

For a Debian installation:

```sh
sudo systemctl status polypack-mcp
sudo systemctl restart polypack-mcp
sudo systemctl stop polypack-mcp
sudo journalctl -u polypack-mcp -n 50 --no-pager
```

For a PyPI user service:

```sh
systemctl --user status polypack-mcp
systemctl --user restart polypack-mcp
journalctl --user -u polypack-mcp -n 50 --no-pager
```

## Store locations

| Installation | Server user | Store |
| --- | --- | --- |
| PyPI setup | Your user | The path passed to `polypack-mcp setup` |
| Debian package | `polypack` | `/var/lib/polypack-mcp` |

Only one server process should open a durable store. Clients communicate with
that process through MCP; they do not access the files directly.

## Port configuration

The default Streamable HTTP port is `8765` and the server binds to localhost by default.

For a PyPI service, choose a port during setup:

```sh
polypack-mcp setup --port 9001 --store ~/.local/share/polypack-mcp
```

For the Debian service, edit `/etc/default/polypack-mcp`:

```sh
POLYPACK_MCP_PORT=9001
```

Then restart the service and update each client URL:

```sh
sudo systemctl restart polypack-mcp
```

The endpoint becomes `http://127.0.0.1:9001/mcp`.

## Concurrency

One shared Streamable HTTP process supports multiple clients. Reads may run concurrently.
Writes are exclusive and are serialized with respect to reads and other
writes. This protects Polypack's in-process graph transaction boundary.

The durable store also has an inter-process lock. Running two independent
server processes against the same store is unsupported.

## Available MCP surface

Tools:

- `memory_store`
- `memory_recall`
- `memory_context`
- `memory_feedback`
- `memory_suppress`
- `memory_supersede`
- `memory_consolidate`
- `memory_link`
- `graph_query`

Resources:

- `polypack://memory/context/{context}`
- `polypack://memory/active`
- `polypack://graph/schema`
- `polypack://stats`
- `polypack://help/workflow`

## Recommended agent workflow

Use `memory_recall` for a targeted question and `memory_context` when assembling
working context. Store durable decisions and outcomes with `memory_store`; use
`procedural` for preferences and conventions, `semantic` for stable facts, and
`episodic` for events or task outcomes.

Link a response, verification, or fix to the memory it addresses with
`memory_link(source_memory_id, target_memory_id, "RESPONDS_TO")`. To follow the
chain, use `memory_recall` with `include_neighbors=true`,
`edge_types=["RESPONDS_TO"]`, and a bounded `depth`.

Context is soft by default. Use `strict_context=true` only when isolation is
required. If recall returns nothing, broaden the query and retry without strict
context before assuming the store is empty.
