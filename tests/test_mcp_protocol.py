import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.shared.memory import create_connected_server_and_client_session
from polypack_mcp.server import create_server


def test_stdio_protocol_lists_surface_and_round_trips_memory():
    async def exercise():
        server = create_server()
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "memory_store", "memory_get", "memory_update", "memory_list_contexts", "memory_delete",
                "memory_recall", "memory_context", "memory_feedback", "memory_suppress",
                "memory_supersede", "memory_consolidate", "memory_link", "memory_unlink",
                "memory_thread", "memory_store_batch", "memory_link_batch",
                "memory_store_with_link", "graph_query",
            }
            store_tool = next(tool for tool in tools.tools if tool.name == "memory_store")
            assert store_tool.inputSchema["properties"]["memory_class"]["enum"] == [
                "entity", "episodic", "procedural", "semantic"
            ]
            stored = await session.call_tool("memory_store", {
                "content": "Polypack MCP protocol works", "context": "test"
            })
            memory = json.loads(stored.content[0].text)
            recalled = await session.call_tool("memory_recall", {
                "query": "MCP protocol", "context": "test"
            })
            recalled_payload = json.loads(recalled.content[0].text)
            assert recalled_payload["items"][0]["id"] == memory["id"]

            linked = await session.call_tool("memory_store", {
                "content": "Follow-up to the protocol test", "context": "test"
            })
            linked_memory = json.loads(linked.content[0].text)
            await session.call_tool("memory_link", {
                "source_memory_id": linked_memory["id"], "target_memory_id": memory["id"],
            })
            neighbor_recall = await session.call_tool("memory_recall", {
                "query": "MCP protocol", "context": "test",
                "include_neighbors": True, "depth": 10,
            })
            neighbor_payload = json.loads(neighbor_recall.content[0].text)
            assert neighbor_payload["metadata"]["depthRequested"] == 10
            assert neighbor_payload["metadata"]["depthApplied"] == 3

            combo = await session.call_tool("memory_store_with_link", {
                "content": "Fix verified for the protocol test", "target_memory_id": memory["id"],
            })
            combo_payload = json.loads(combo.content[0].text)
            assert combo_payload["link"] == {
                "source": combo_payload["memory"]["id"], "type": "RESPONDS_TO", "target": memory["id"],
            }
            combo_neighbor_recall = await session.call_tool("memory_recall", {
                "query": "MCP protocol", "context": "test",
                "include_neighbors": True, "depth": 1, "neighbor_limit": 5,
            })
            combo_neighbor_payload = json.loads(combo_neighbor_recall.content[0].text)
            item_ids = {item["id"] for item in combo_neighbor_payload["items"]}
            assert combo_payload["memory"]["id"] in item_ids

            help_resource = await session.read_resource("polypack://help/workflow")
            assert "memory_link" in help_resource.contents[0].text
            assert "memory_store_with_link" in help_resource.contents[0].text
            stats = await session.read_resource("polypack://stats")
            assert '"memories": 3' in stats.contents[0].text

    asyncio.run(exercise())


def test_external_stdio_process_initializes():
    process = subprocess.Popen(
        [sys.executable, "-m", "polypack_mcp.server"],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.5)
        assert process.poll() is None
    finally:
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_streamable_http_supports_oauth_registration_pkce_and_slashless_mcp():
    async def exercise():
        server = create_server()
        app = server.streamable_http_app()
        verifier = "jan-pkce-code-verifier-0123456789abcdef"
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .decode("ascii")
            .rstrip("=")
        )
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "jan", "version": "1.0"},
            },
        }
        mcp_headers = {"Accept": "application/json, text/event-stream"}

        async with server.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
                as_meta = await client.get("/.well-known/oauth-authorization-server")
                assert as_meta.status_code == 200
                assert as_meta.json()["registration_endpoint"] == "http://127.0.0.1:8765/register"
                assert as_meta.json()["authorization_endpoint"] == "http://127.0.0.1:8765/authorize"
                assert as_meta.json()["token_endpoint"] == "http://127.0.0.1:8765/token"

                pr_meta = await client.get("/.well-known/oauth-protected-resource")
                assert pr_meta.status_code == 200
                assert pr_meta.json()["authorization_servers"] == ["http://127.0.0.1:8765"]

                reg = await client.post(
                    "/register",
                    json={
                        "client_name": "Jan",
                        "redirect_uris": ["http://127.0.0.1:54321/callback"],
                        "grant_types": ["authorization_code"],
                        "response_types": ["code"],
                        "token_endpoint_auth_method": "none",
                    },
                )
                assert reg.status_code == 201
                client_id = reg.json()["client_id"]
                assert client_id

                # Verify ephemeral loopback port redirect URI is accepted (RFC 8252).
                bad_auth = await client.get(
                    "/authorize",
                    params={
                        "response_type": "code",
                        "client_id": client_id,
                        "redirect_uri": "http://127.0.0.1:54322/callback",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                        "state": "state-bad",
                    },
                    follow_redirects=False,
                )
                assert bad_auth.status_code == 302
                bad_location = urlparse(bad_auth.headers["location"])
                assert bad_location.netloc == "127.0.0.1:54322"
                assert parse_qs(bad_location.query)["state"] == ["state-bad"]
                bad_code = parse_qs(bad_location.query)["code"][0]

                wrong_token = await client.post(
                    "/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": bad_code,
                        "client_id": client_id,
                        "redirect_uri": "http://127.0.0.1:54322/callback",
                        "code_verifier": "wrong-verifier",
                    },
                )
                assert wrong_token.status_code == 400
                assert wrong_token.json()["error"] == "invalid_grant"

                auth = await client.get(
                    "/authorize",
                    params={
                        "response_type": "code",
                        "client_id": client_id,
                        "redirect_uri": "http://127.0.0.1:54323/callback",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                        "state": "state-ok",
                    },
                    follow_redirects=False,
                )
                assert auth.status_code == 302
                location = urlparse(auth.headers["location"])
                assert location.netloc == "127.0.0.1:54323"
                query = parse_qs(location.query)
                assert query["state"] == ["state-ok"]
                code = query["code"][0]

                token = await client.post(
                    "/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "client_id": client_id,
                        "redirect_uri": "http://127.0.0.1:54323/callback",
                        "code_verifier": verifier,
                    },
                )
                assert token.status_code == 200
                token_body = token.json()
                assert token_body["token_type"] == "Bearer"
                assert token_body["access_token"]
                assert token_body["refresh_token"]

                refreshed = await client.post(
                    "/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": token_body["refresh_token"],
                        "client_id": client_id,
                    },
                )
                assert refreshed.status_code == 200
                refreshed_body = refreshed.json()
                assert refreshed_body["access_token"]
                assert refreshed_body["access_token"] != token_body["access_token"]

                for path in ("/mcp", "/mcp/"):
                    unauth_mcp = await client.post(path, headers=mcp_headers, json=init_payload, follow_redirects=False)
                    assert unauth_mcp.status_code == 200
                    auth_mcp = await client.post(
                        path,
                        headers={**mcp_headers, "Authorization": f"Bearer {refreshed_body['access_token']}"},
                        json=init_payload,
                        follow_redirects=False,
                    )
                    assert auth_mcp.status_code == 200

    asyncio.run(exercise())
