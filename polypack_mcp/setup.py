"""Local service setup for sharing one Polypack server between MCP clients."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def install_user_service(store: str, port: int = 8765, start: bool = True,
                         embedding_url: str | None = None) -> Path:
    """Install a systemd user service and optionally start it."""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise RuntimeError("systemctl was not found; use --print-config or start the HTTP server manually")

    executable = shutil.which("polypack-mcp")
    if not executable:
        raise RuntimeError("polypack-mcp executable was not found on PATH")

    store_path = Path(store).expanduser().resolve()
    store_path.mkdir(parents=True, exist_ok=True)
    unit_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "polypack-mcp.service"
    unit_path.write_text(
        (
            "[Unit]\n"
            "Description=Polypack MCP shared memory server\n"
            "After=network.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            "ExecStart={executable} --transport streamable-http --port {port} --store {store}\n"
            "{embedding_environment}"
            "Restart=on-failure\n"
            "RestartSec=3\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        ).format(
            executable=executable, port=port, store=store_path,
            embedding_environment=(f"Environment=POLYPACK_MCP_EMBEDDING_URL={embedding_url}\n"
                                   if embedding_url else ""),
        ),
        encoding="utf-8",
    )
    subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
    if start:
        subprocess.run([systemctl, "--user", "enable", "--now", "polypack-mcp.service"], check=True)
    return unit_path


def print_client_config(port: int = 8765) -> None:
    url = f"http://127.0.0.1:{port}/mcp/"
    print("Shared Polypack MCP endpoint:")
    print(url)
    print("\nCodex (~/.codex/config.toml):")
    print("[mcp_servers.polypack]")
    print(f'url = "{url}"')
    print("\nClaude Desktop configuration:")
    print('{"mcpServers": {"polypack": {"url": "' + url + '"}}}')
