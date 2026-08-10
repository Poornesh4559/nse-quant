"""MCP test client: connect to the nse-quant server over stdio and call tools.
Usage: python -m mcp_nse.client <tool_name> '<json-args>' [tool2 '<json>']
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    calls = sys.argv[1:] or ["get_sentiment", '{"symbol": "TCS"}']
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_nse.server"],
        cwd="/home/ubuntu/nse-quant",
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"SERVER OK — {len(tools.tools)} tools exposed:")
            print("  " + ", ".join(t.name for t in tools.tools))
            i = 0
            while i < len(calls):
                name = calls[i]
                args = json.loads(calls[i + 1]) if i + 1 < len(calls) else {}
                i += 2
                print(f"\n>>> {name}{args}")
                res = await session.call_tool(name, args)
                for c in res.content:
                    print(c.text[:1200])


if __name__ == "__main__":
    asyncio.run(main())
