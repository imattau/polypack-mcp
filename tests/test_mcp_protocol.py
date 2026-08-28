import asyncio
import json
import subprocess
import sys
import time

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
