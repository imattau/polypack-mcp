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

For repeatable installation and upgrades, use the public APT repository instead
of downloading an individual release asset:

```sh
curl -fsSL https://imattau.github.io/polypack-mcp/gpg.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/polypack-mcp.gpg
echo "deb [signed-by=/usr/share/keyrings/polypack-mcp.gpg] https://imattau.github.io/polypack-mcp stable main" \
  | sudo tee /etc/apt/sources.list.d/polypack-mcp.list
sudo apt update
sudo apt install polypack-mcp
```

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

Hermes Agent:

Hermes has a native MCP client. For a dedicated Polypack process, add this to
`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  polypack:
    command: uvx
    args:
      - --from
      - polypack-mcp[polypack]
      - polypack-mcp
      - --store
      - /home/user/.local/share/polypack-mcp
```

Alternatively, after installing the package locally, the equivalent CLI
command is:

```sh
hermes mcp add polypack --command polypack-mcp --args --store /home/user/.local/share/polypack-mcp
```

If Hermes should share the same memory with Codex or Claude, run
`polypack-mcp setup --store ~/.local/share/polypack-mcp` and configure Hermes
with the shared endpoint instead:

```yaml
mcp_servers:
  polypack:
    url: http://127.0.0.1:8765/mcp/
```

Restart Hermes after changing the configuration. Its Polypack tools will be
available with the `mcp_polypack_` prefix.

When using the shared Streamable HTTP service, do not also configure clients with a
`command` and the same `--store` path. That would start competing server
processes.

## RPM installation

On Fedora, RHEL-compatible, or other RPM-based systems:

```sh
sudo rpm --import https://imattau.github.io/polypack-mcp/rpm/RPM-GPG-KEY-polypack-mcp
sudo tee /etc/yum.repos.d/polypack-mcp.repo >/dev/null <<'EOF'
[polypack-mcp]
name=Polypack MCP
baseurl=https://imattau.github.io/polypack-mcp/rpm/
enabled=1
gpgcheck=1
gpgkey=https://imattau.github.io/polypack-mcp/rpm/RPM-GPG-KEY-polypack-mcp
EOF
sudo dnf install polypack-mcp
```

The RPM package installs the same system-level service as the Debian package.
The matching release asset can also be installed directly:

```sh
sudo dnf install ./polypack-mcp-<version>-1.x86_64.rpm
```
