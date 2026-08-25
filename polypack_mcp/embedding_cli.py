"""Lifecycle commands for the optional managed embedding helper."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .embeddings import HTTPEmbeddingProvider


DEFAULT_MODEL = "qwen3"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_PORT = 8766


def _paths(store: Path) -> tuple[Path, Path, Path]:
    root = store / ".embedding"
    return root, root / "venv", root / "cache"


def _unit_path(system: bool) -> Path:
    return Path("/etc/systemd/system" if system else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user") / "polypack-mcp-embedding.service"


def _main_dropin(system: bool) -> Path:
    base = Path("/etc/systemd/system" if system else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user")
    return base / "polypack-mcp.service.d" / "embedding.conf"


def _systemctl(system: bool, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["systemctl"]
    if not system:
        command.append("--user")
    command.extend(args)
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _write_unit(store: Path, venv: Path, cache: Path, system: bool) -> None:
    unit = _unit_path(system)
    unit.parent.mkdir(parents=True, exist_ok=True)
    package_root = str(Path(__file__).resolve().parent.parent)
    venv_site = subprocess.check_output(
        [str(venv / "bin/python"), "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()
    service_user = "User=polypack\nGroup=polypack\n" if system else ""
    unit.write_text(
        "[Unit]\n"
        "Description=Polypack Qwen embedding helper\n"
        "After=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"{service_user}"
        f"Environment=HF_HOME={cache}\n"
        f"Environment=PYTHONPATH={package_root}:{venv_site}\n"
        f"ExecStart={venv / 'bin/python'} -m polypack_mcp.embeddings --host 127.0.0.1 --port {EMBEDDING_PORT}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n",
        encoding="utf-8",
    )


def _configure_main(system: bool, enabled: bool) -> None:
    dropin = _main_dropin(system)
    if enabled:
        dropin.parent.mkdir(parents=True, exist_ok=True)
        dropin.write_text(
            "[Service]\nEnvironment=POLYPACK_MCP_EMBEDDING_URL=http://127.0.0.1:8766\n",
            encoding="utf-8",
        )
    elif dropin.exists():
        dropin.unlink()


def _wait_for_helper(provider: HTTPEmbeddingProvider, timeout: int = 900) -> dict:
    deadline = time.time() + timeout
    last_error = "not started"
    while time.time() < deadline:
        try:
            return provider.health()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    raise RuntimeError(f"embedding helper did not become ready: {last_error}")


def _reindex(store: Path, provider: HTTPEmbeddingProvider) -> int:
    from polypack import PolyGraph

    graph = PolyGraph.open(store)
    changed = 0
    try:
        nodes = list(graph._nodes.values())
        for start in range(0, len(nodes), 32):
            batch = nodes[start:start + 32]
            texts = []
            for node in batch:
                data = node.get("data", {})
                context = data.get("context")
                content = data.get("content", "")
                texts.append(f"context: {context}\ncontent: {content}" if context else content)
            vectors = provider.embed(texts)
            for node, vector in zip(batch, vectors):
                graph.update_node(node["id"], data={}, vector=vector)
                changed += 1
        graph.checkpoint()
    finally:
        graph.close_store()
    return changed


def _reindex_with_service_ownership(store: Path, provider: HTTPEmbeddingProvider, system: bool) -> int:
    """Reindex and leave a system store writable by the MCP service user.

    System setup/reindex is normally invoked with sudo, so checkpoint files
    created by Polypack can otherwise become root-owned. The main service runs
    as ``polypack`` and must be able to append to the WAL afterward.
    """
    try:
        return _reindex(store, provider)
    finally:
        if system:
            subprocess.run(["chown", "-R", "polypack:polypack", str(store)], check=True)


def setup_qwen(store: Path, system: bool) -> None:
    if shutil.which("systemctl") is None:
        raise RuntimeError("systemd is required for managed embeddings; use an external provider instead")
    root, venv, cache = _paths(store)
    root.mkdir(parents=True, exist_ok=True)
    if not (venv / "bin/python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run(
        [str(venv / "bin/pip"), "install", "--upgrade", "sentence-transformers>=3.0,<6"],
        check=True,
    )
    if system:
        subprocess.run(["chown", "-R", "polypack:polypack", str(root)], check=True)
    _write_unit(store, venv, cache, system)
    _configure_main(system, True)
    _systemctl(system, "daemon-reload")
    _systemctl(system, "enable", "--now", "polypack-mcp-embedding.service")
    provider = HTTPEmbeddingProvider(f"http://127.0.0.1:{EMBEDDING_PORT}", timeout=60)
    health = _wait_for_helper(provider)
    print(f"Embedding helper ready: {health['model']} ({health['dimension']} dimensions)")
    main_was_active = _systemctl(system, "is-active", "--quiet", "polypack-mcp.service", check=False).returncode == 0
    if main_was_active:
        _systemctl(system, "stop", "polypack-mcp.service")
    try:
        from polypack import PolyGraph  # validates the installed Polypack backend before stopping work
        del PolyGraph
        count = _reindex_with_service_ownership(store, provider, system)
        print(f"Reindexed {count} memories")
    finally:
        if main_was_active:
            _systemctl(system, "start", "polypack-mcp.service")
    print("Semantic retrieval is enabled.")


def disable(store: Path, system: bool) -> None:
    _configure_main(system, False)
    if shutil.which("systemctl"):
        _systemctl(system, "disable", "--now", "polypack-mcp-embedding.service", check=False)
        _systemctl(system, "daemon-reload", check=False)
        _systemctl(system, "restart", "polypack-mcp.service", check=False)
    print("Semantic retrieval disabled; the model cache was retained.")


def reindex(store: Path, system: bool) -> None:
    provider = HTTPEmbeddingProvider(f"http://127.0.0.1:{EMBEDDING_PORT}", timeout=60)
    health = _wait_for_helper(provider)
    print(f"Embedding helper ready: {health['model']}")
    main_was_active = _systemctl(system, "is-active", "--quiet", "polypack-mcp.service", check=False).returncode == 0
    if main_was_active:
        _systemctl(system, "stop", "polypack-mcp.service")
    try:
        count = _reindex_with_service_ownership(store, provider, system)
        print(f"Reindexed {count} memories")
    finally:
        if main_was_active:
            _systemctl(system, "start", "polypack-mcp.service")


def status() -> None:
    provider = HTTPEmbeddingProvider(f"http://127.0.0.1:{EMBEDDING_PORT}", timeout=3)
    try:
        print(json.dumps(provider.health(), indent=2))
    except Exception as exc:
        print(json.dumps({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}, indent=2))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Manage optional Polypack embeddings")
    parser.add_argument("action", choices=("setup", "status", "reindex", "disable"))
    parser.add_argument("model", nargs="?", choices=(DEFAULT_MODEL,))
    parser.add_argument("--store", default="~/.local/share/polypack-mcp")
    parser.add_argument("--system", action="store_true", help="manage system services instead of user services")
    args = parser.parse_args(argv)
    store = Path(args.store).expanduser().resolve()
    if args.action == "setup":
        if args.model != DEFAULT_MODEL:
            parser.error("setup currently requires the qwen3 model")
        setup_qwen(store, args.system)
    elif args.action == "status":
        status()
    elif args.action == "disable":
        disable(store, args.system)
    else:
        reindex(store, args.system)
    return 0
