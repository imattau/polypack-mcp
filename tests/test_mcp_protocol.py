import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from polypack_mcp.server import create_server


def test_stdio_protocol_lists_surface_and_round_trips_memory():
    async def exercise():
        server = create_server()
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "memory_store", "memory_recall", "memory_context", "memory_feedback",
                "memory_suppress", "memory_supersede", "memory_consolidate", "graph_query",
            }
            stored = await session.call_tool("memory_store", {
                "content": "Polypack MCP protocol works", "context": "test"
            })
            memory = json.loads(stored.content[0].text)
            recalled = await session.call_tool("memory_recall", {
                "query": "MCP protocol", "context": "test"
            })
            recalled_payload = json.loads(recalled.content[0].text)
            assert recalled_payload["items"][0]["id"] == memory["id"]
            stats = await session.read_resource("polypack://stats")
            assert '"memories": 1' in stats.contents[0].text

    asyncio.run(exercise())


def test_external_stdio_process_initializes():
    async def exercise():
        params = StdioServerParameters(command=sys.executable, args=["-m", "polypack_mcp.server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert len(tools.tools) == 8

    asyncio.run(exercise())
