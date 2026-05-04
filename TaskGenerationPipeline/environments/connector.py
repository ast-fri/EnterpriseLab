"""
Universal connector for managing multiple environment adapters
"""
from typing import List, Dict, Any, Optional, Type
from .base import EnvironmentAdapter
from .tool_wrapper import ToolWrapper


class UniversalConnector:
    """
    Manages multiple environment adapters
    Single interface for AutoQuest to interact with all environments
    """
    
    def __init__(self, gpt_caller=None):
        """
        Args:
            gpt_caller: GPT caller instance (for environments that need it)
        """
        self.gpt_caller = gpt_caller
        self.adapters: List[EnvironmentAdapter] = []
        self.all_tools: List[ToolWrapper] = []
        self.tool_to_env_map: Dict[str, str] = {}
    
    def add_environment(
        self, 
        adapter_class: Type[EnvironmentAdapter],
        config: Dict[str, Any],
        auto_connect: bool = True
    ) -> bool:
        """
        Add an environment adapter
        
        Args:
            adapter_class: The adapter class (e.g., CRMArenaAdapter)
            config: Configuration dict for the adapter
            auto_connect: If True, automatically connect and load tools
        
        Returns:
            True if successful, False otherwise
        """
        try:
            adapter = adapter_class(config)
            
            if auto_connect:
                if adapter.connect():
                    tools = adapter.load_tools()
                    
                    self.adapters.append(adapter)
                    self.all_tools.extend(tools)
                    
                    for tool in tools:
                        self.tool_to_env_map[tool.name] = adapter.env_name
                    
                    print(f"✅ Added {adapter.env_name}: {len(tools)} tools")
                    return True
                else:
                    print(f"❌ Failed to connect {adapter.env_name}")
                    return False
            else:
                self.adapters.append(adapter)
                return True
                
        except Exception as e:
            print(f"❌ Error adding environment: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_all_tools(self) -> List[ToolWrapper]:
        """Get all tools from all environments"""
        return self.all_tools
    
    def get_tools_by_environment(self, env_name: str) -> List[ToolWrapper]:
        """Get tools from specific environment"""
        return [
            tool for tool in self.all_tools 
            if self.tool_to_env_map.get(tool.name) == env_name
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all connected environments"""
        summary = {
            "total_environments": len(self.adapters),
            "total_tools": len(self.all_tools),
            "environments": {}
        }
        
        for adapter in self.adapters:
            metadata = adapter.get_metadata()
            summary["environments"][adapter.env_name] = metadata
        
        return summary
    
    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Invoke a tool by name
        
        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        tool = next((t for t in self.all_tools if t.name == tool_name), None)
        
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        return tool.invoke(**kwargs)
    
    def cleanup(self):
        """Cleanup all adapters"""
        print("\n🔌 Cleaning up connections...")
        for adapter in self.adapters:
            try:
                adapter.disconnect()
            except Exception as e:
                print(f"⚠️  Error disconnecting {adapter.env_name}: {e}")
        
        print("✅ Cleanup complete")
