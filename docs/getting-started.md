# Getting started

Polypack MCP exposes persistent adaptive memory to MCP clients such as Codex
and Claude. Choose one installation path, then configure clients to connect to
the server.

## PyPI installation

Install the server and the durable Polypack backend:

```sh
python3 -m pip install 'polypack-mcp[polypack]'
```

For a single client using an isolated process, configure that client to launch
`polypack-mcp` with stdio. For shared memory across multiple clients, create a
long-running Streamable HTTP service:

```sh
polypack-mcp setup --store ~/.local/share/polypack-mcp
```

The default endpoint is:

```text
http://127.0.0.1:8765/mcp/
```

The setup command creates a `systemd --user` service, enables it, starts it,
and prints client configuration snippets. It does not edit client
configuration files automatically.

## Debian installation

Download the `.deb` asset from the [GitHub release](https://github.com/imattau/polypack-mcp/releases),
then install it:

```sh
sudo apt install ./polypack-mcp_<version>_amd64.deb
```

The Debian package installs and starts a system service under the dedicated
`polypack` user. Its default store is `/var/lib/polypack-mcp` and its endpoint
is `http://127.0.0.1:8765/mcp/`.

The packaged native dependencies target Python 3.12, so the package declares
`python3 (>= 3.12)`.

## Client configuration

Use the Streamable HTTP URL in every client that should share the same memory.

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.polypack]
url = "http://127.0.0.1:8765/mcp/"
startup_timeout_sec = 30
```

Claude Desktop:

```json
{
  "mcpServers": {
    "polypack": {
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

When using the shared Streamable HTTP service, do not also configure clients with a
`command` and the same `--store` path. That would start competing server
processes.

## RPM installation

On Fedora, RHEL-compatible, or other RPM-based systems:

```sh
sudo dnf install ./polypack-mcp-<version>-1.x86_64.rpm
```

The RPM package installs the same system-level service as the Debian package.
