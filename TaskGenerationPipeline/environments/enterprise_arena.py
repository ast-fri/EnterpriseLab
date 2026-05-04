"""
EnterpriseArena environment adapter with MCP servers
Your existing MCP-based environment
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from .base import EnvironmentAdapter
from .tool_wrapper import ToolWrapper


class EnterpriseArenaAdapter(EnvironmentAdapter):
    """
    Adapter for EnterpriseArena (your existing MCP servers)
    Uses langchain_mcp_adapters to load tools from multiple MCP servers
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                "mcp_config_path": "./mcp_config_http.json",
                "connection_timeout": 15.0,
                "tool_load_timeout": 10.0
            }
        """
        super().__init__(config)
        
        self.mcp_config_path = config.get("mcp_config_path", "./mcp_config_http.json")
        self.connection_timeout = config.get("connection_timeout", 15.0)
        self.tool_load_timeout = config.get("tool_load_timeout", 10.0)
        
        self.mcp_servers_config = {}
        self.mcp_client = None
        self.sessions = {}
        self.stack = None
        
        self.env_name = "EnterpriseArena"
    
    def connect(self) -> bool:
        """
        Load MCP config (actual connection happens in load_tools)
        MCP connections need async context, so we defer to load_tools
        """
        try:
            import json
            
            # Load MCP server configuration
            with open(self.mcp_config_path, 'r') as f:
                config_data = json.load(f)
            
            self.mcp_servers_config = config_data.get("mcpServers", {})
            
            print(f"✅ Loaded MCP configuration")
            print(f"   • MCP servers configured: {len(self.mcp_servers_config)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load MCP config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tools(self) -> List[ToolWrapper]:
        """
        Connect to MCP servers and load tools
        This is async, so we use asyncio.run
        """
        if not self.mcp_servers_config:
            raise RuntimeError("Must call connect() first!")
        
        try:
            print(f"\n🔧 Loading EnterpriseArena MCP tools...")
            
            # Run async connection in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                import nest_asyncio
                nest_asyncio.apply()
            
            loop.run_until_complete(self._async_load_tools())
            
            print(f"✅ Loaded {len(self.tools)} EnterpriseArena MCP tools")
            return self.tools
            
        except Exception as e:
            print(f"❌ Failed to load EnterpriseArena tools: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _async_load_tools(self):
        """Async method to load tools from MCP servers"""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
        
        # Create MCP client
        self.mcp_client = MultiServerMCPClient(self.mcp_servers_config)
        
        # Use AsyncExitStack for persistent connections
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        
        # Connect to each MCP server
        for server_name in self.mcp_servers_config.keys():
            try:
                print(f"   🔄 Connecting to {server_name}...")
                
                # Establish persistent session with timeout
                session = await asyncio.wait_for(
                    self.stack.enter_async_context(self.mcp_client.session(server_name)),
                    timeout=self.connection_timeout
                )
                
                # Store session
                self.sessions[server_name] = session
                
                # Load tools from this server
                server_tools = await asyncio.wait_for(
                    load_mcp_tools(session),
                    timeout=self.tool_load_timeout
                )
                
                # Convert each MCP tool to ToolWrapper
                for mcp_tool in server_tools:
                    tool = self._wrap_mcp_tool(mcp_tool, server_name)
                    # print(f"{tool.name} : {tool.args_schema}")
                    if tool:
                        self.tools.append(tool)
                
                print(f"   ✓ {server_name}: {len(server_tools)} tools loaded")
            
            except asyncio.TimeoutError:
                print(f"   ⚠️  {server_name}: Connection timeout (skipping)")
                continue
            
            except Exception as e:
                print(f"   ✗ {server_name}: {str(e)}")
                continue
    
    def _wrap_mcp_tool(self, mcp_tool, server_name: str) -> Optional[ToolWrapper]:
        """
        Wrap MCP tool into ToolWrapper
        Extracts schema from LangChain MCP tool
        """
        try:
            # Extract name and description
            original_name = mcp_tool.name
            name = f"{server_name}_{original_name}"
            description = mcp_tool.description or f"Tool from {server_name}"
            args_schema = mcp_tool.args_schema
            
            # Return schema (MCP doesn't provide this, so we infer)
            return_schema = {
                "result": f"Result from {server_name} {original_name} tool"
            }
            
            # ✅ Create ASYNC executor that calls MCP tool with ainvoke
            async def tool_executor(**kwargs) -> Any:
                """Execute MCP tool asynchronously"""
                try:
                    # MCP tools from langchain_mcp_adapters are async-only
                    # Use ainvoke instead of invoke/run
                    if hasattr(mcp_tool, 'ainvoke'):
                        result = await mcp_tool.ainvoke(kwargs)
                    elif hasattr(mcp_tool, 'arun'):
                        result = await mcp_tool.arun(**kwargs)
                    else:
                        # Fallback: try calling directly as async
                        result = await mcp_tool(**kwargs)
                    
                    return result
                    
                except Exception as e:
                    return {"error": str(e), "success": False}
            
            # Create ToolWrapper with async executor
            tool = ToolWrapper(
                name=name,
                description=description,
                invoke_fn=tool_executor,  # Now async!
                args_schema=args_schema,
                return_schema=return_schema,
                metadata={
                    "env": "enterprise_arena",
                    "server": server_name,
                    "original_name": original_name,
                    "mcp_tool": True
                }
            )
            
            return tool
            
        except Exception as e:
            print(f"⚠️  Failed to wrap MCP tool {mcp_tool.name}: {e}")
            return None

    
    def disconnect(self):
        """Cleanup MCP connections"""
        try:
            if self.stack:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.stack.__aexit__(None, None, None))
            
            self.sessions = {}
            self.mcp_client = None
            print(f"🔌 Disconnected from EnterpriseArena MCP servers")
            
        except Exception as e:
            print(f"⚠️  Error disconnecting: {e}")
