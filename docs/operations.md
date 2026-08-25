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

The endpoint becomes `http://127.0.0.1:9001/mcp/`.

## Optional embeddings

Enable managed Qwen semantic retrieval with:

```sh
sudo polypack-mcp embeddings setup qwen3 --system --store /var/lib/polypack-mcp
```

The helper listens only on `127.0.0.1:8766`. The MCP sends provider-generated
vectors to Polypack, which owns vector persistence and similarity search. The
base service continues to operate with lexical and activation fallback if the
helper is stopped. Inspect the provider with:

```sh
polypack-mcp embeddings status
```

### Resource footprint and idle unload

Qwen3-Embedding-0.6B loads in bfloat16, using roughly 1GB resident once a
request has triggered the load (fp32 would be ~2.4GB — just the weights at 4
bytes/param). A background watchdog unloads the model after 15 minutes with
no `/embed` or `/health` calls, dropping the helper's footprint to a few MB;
the next request reloads it in a few seconds (weights stay cached under
`HF_HOME`, so this is a re-instantiate, not a re-download). The idle window
is not exposed through `embeddings setup` — override it by editing
`ExecStart` in the unit (`polypack-mcp-embedding.service` under
`/etc/systemd/system` or the user equivalent) to add `--idle-timeout SECONDS`
(`0` disables unloading), then `systemctl restart polypack-mcp-embedding`.

### Score components

When the helper is reachable, `memory_recall`/`memory_context` weight results
as `0.55 * semantic + 0.30 * lexical + 0.15 * activation`, and
`scoreComponents` reports all three (summing to `score`). Without a reachable
helper, scoring falls back to `lexical + 0.25 * activation` and
`scoreComponents` omits `semantic` entirely — its absence is itself a signal
that recall is running lexical-only.

This weighting is `polypack-mcp`'s retrieval policy, not part of `polypack`
itself. The core library supplies the primitives — vector similarity, graph
traversal, activation decay — and this server chooses how to blend them into
one ranked result; a different polypack-based consumer could reasonably pick
different weights. Don't expect to find this formula in `polypack` core.

## Concurrency

One shared stateless Streamable HTTP process supports multiple clients without
per-client MCP session state. Reads may run concurrently.
Writes are exclusive and are serialized with respect to reads and other
writes. This protects Polypack's in-process graph transaction boundary.

The durable store also has an inter-process lock. Running two independent
server processes against the same store is unsupported.

## Available MCP surface

Tools:

- `memory_store`
- `memory_get`
- `memory_update`
- `memory_list_contexts`
- `memory_delete`
- `memory_recall`
- `memory_context`
- `memory_feedback`
- `memory_suppress`
- `memory_supersede`
- `memory_consolidate`
- `memory_link`
- `memory_unlink`
- `memory_thread`
- `memory_store_batch`
- `memory_link_batch`
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
`edge_types=["RESPONDS_TO"]`, a bounded `depth`, and an explicit
`neighbor_limit`. The overall `limit` includes both primary and neighbor
items; `moreNeighborsAvailable` reports when the neighbor bound truncated
eligible results. Graph edges are authoritative for relationships; use
`graph_query(operation="relationship_diagnostics")` to audit legacy
`provenance.responds_to` values.

Use `memory_get` for exact lookup and `memory_update` for context, confidence,
provenance, or metadata changes. Content changes should use supersession.
`memory_delete` permanently removes a memory and requires `confirm=true`; use
`memory_suppress` when history should be retained. `memory_unlink` removes
directed graph edges between two memories.

Context is soft by default. Use `strict_context=true` only when isolation is
required. If recall returns nothing, broaden the query and retry without strict
context before assuming the store is empty.
