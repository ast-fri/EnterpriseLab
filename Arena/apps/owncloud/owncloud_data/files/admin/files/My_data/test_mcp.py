#!/usr/bin/env python3
import asyncio
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

server = Server("test-mcp")

@server.list_tools()
async def handle_list_tools():
    return [
        types.Tool(
            name="hello",
            description="Say hello",
            inputSchema={"type": "object", "properties": {}}
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments):
    return [types.TextContent(type="text", text="Hello from MCP!")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())