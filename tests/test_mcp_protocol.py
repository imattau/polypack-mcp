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
                "memory_store", "memory_recall", "memory_context", "memory_feedback",
                "memory_suppress", "memory_supersede", "memory_consolidate", "memory_link",
                "memory_thread", "memory_store_batch", "memory_link_batch", "graph_query",
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
            help_resource = await session.read_resource("polypack://help/workflow")
            assert "memory_link" in help_resource.contents[0].text
            stats = await session.read_resource("polypack://stats")
            assert '"memories": 1' in stats.contents[0].text

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
