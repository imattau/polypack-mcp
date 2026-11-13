# Troubleshooting

## Service is restarting with exit status 1

Read the service log first:

```sh
sudo journalctl -u polypack-mcp -n 80 --no-pager
```

For a Debian installation, verify the service user owns the store:

```sh
sudo chown -R polypack:polypack /var/lib/polypack-mcp
sudo systemctl restart polypack-mcp
```

## NumPy or native extension import errors

The Debian package contains native Python dependencies and targets Python
3.12. Confirm the runtime:

```sh
/usr/bin/python3 --version
```

On an older distribution, use the PyPI installation in a matching virtual
environment instead of the Debian package.

## Port is already in use

Check the configured port:

```sh
cat /etc/default/polypack-mcp
```

Change `POLYPACK_MCP_PORT`, restart the service, and update the client URL.

For a PyPI setup, recreate the user service with a different `--port` value.

## Claude or Codex cannot load the server

Check that the client uses the Streamable HTTP URL rather than launching another process:

```text
http://127.0.0.1:8765/mcp/
```

If the client configuration contains both `command` and `--store`, remove that
entry and replace it with the URL configuration. Restart the client after
changing its MCP settings.

## Jan OAuth sign-in or 404 on `/register`

Older `polypack-mcp` releases did not expose OAuth discovery or Dynamic Client
Registration endpoints on the local loopback server, causing Jan's browser
sign-in flow to fail with `POST http://127.0.0.1:8765/register: HTTP 404 Not Found`.
Current releases expose loopback OAuth 2.1 discovery, `/register`, `/authorize`,
and `/token` (with PKCE) alongside the unauthenticated Streamable HTTP endpoint
at both `http://127.0.0.1:8765/mcp/` and `http://127.0.0.1:8765/mcp`. Upgrade and
restart `polypack-mcp`, or configure Jan directly with
`jan cli mcp add polypack-mcp --type http --url http://127.0.0.1:8765/mcp/ --active`.

## Store lock errors

A second server process may already be using the store. Check the service
first, then inspect the lock file:

```sh
systemctl status polypack-mcp
ls -l /var/lib/polypack-mcp/store.lock
```

Do not remove a lock while a Polypack server is running. If the lock belongs to
a crashed process and no server is active, stop the service, remove the stale
lock, and start the service again.

## Embedding setup fails

The optional Qwen setup needs `systemd`, Python virtual-environment support,
network access for the first model download, and approximately 2 GB of free
disk space. Check the helper state with:

```sh
polypack-mcp embeddings status
sudo systemctl status polypack-mcp-embedding
```

If the helper is unavailable, the main MCP service should still answer using
lexical and activation retrieval. Retry setup after fixing the dependency or
network problem; it is safe to run repeatedly.
