import asyncio
import threading
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import time

class EmailMCPClient:
    """MCP Client for Email Server with proper async handling for Gradio"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.tools = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._connected = False
    
    def _start_background_loop(self):
        """Start a background event loop in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def ensure_loop(self):
        """Ensure the event loop is running"""
        if self.thread is None:
            self.thread = threading.Thread(target=self._start_background_loop, daemon=True)
            self.thread.start()
            # Wait for loop to be ready
            while self.loop is None:
                time.sleep(0.01)
    
    
    def connect_to_server(self, server_script_path: str = "server.py"):
        """Connect to the Email MCP server"""
        if self._connected:
            return f"Already connected with {len(self.tools)} tools"
        
        self.ensure_loop()
        
        # Define the async connection function
        async def do_connect():
            self.exit_stack = AsyncExitStack()
            
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
                env=None
            )
            
            # Establish connection
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            self.tools = response.tools
            self._connected = True
            
            return len(self.tools)
        
        # Run it in the event loop and get the result
        future = asyncio.run_coroutine_threadsafe(do_connect(), self.loop)
        num_tools = future.result(timeout=30)
        return f"✅ Connected to Email Server successfully!\n📦 {num_tools} tools available"
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        if not self._connected:
            raise Exception("Not connected to server. Call connect_to_server() first.")
        
        # Define the async call function
        async def do_call():
            result = await self.session.call_tool(tool_name, arguments)
            return result
        
        # Run it in the event loop
        future = asyncio.run_coroutine_threadsafe(do_call(), self.loop)
        return future.result(timeout=30)
    
    def disconnect(self):
        """Disconnect from the server"""
        if not self._connected:
            return
        
        async def do_disconnect():
            if self.exit_stack:
                await self.exit_stack.aclose()
                self._connected = False
        
        future = asyncio.run_coroutine_threadsafe(do_disconnect(), self.loop)
        future.result(timeout=10)


# Global client instance
email_client = EmailMCPClient()
