"""
Tau-bench environment adapter with ToolWrapper
Supports both retail and airline domains
"""
import os
import sys
import json
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from dotenv import load_dotenv
import importlib.util

from .base import EnvironmentAdapter
from .tool_wrapper import ToolWrapper


class TauBenchAdapter(EnvironmentAdapter):
    """
    Adapter for Tau-bench (retail or airline domain)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                "domain": "retail" | "airline",
                "tau_bench_path": "./environments/tau-bench",
                "data_path": None,  # Auto-detected if None
                "tools_path": None  # Auto-detected if None
            }
        """
        super().__init__(config)
        
        self.domain = config.get("domain", "retail")
        self.tau_bench_path = config.get("tau_bench_path", "./environments/tau-bench")
        
        # Set paths
        self.base_path = Path(self.tau_bench_path)
        self.env_path = self.base_path / "tau_bench" / "envs" / self.domain
        
        self.data_path = config.get("data_path") or (self.env_path / "data")
        self.tools_path = config.get("tools_path") or (self.env_path / "tools")
        
        # Add tau-bench to Python path
        tau_bench_root = str(self.base_path)
        if tau_bench_root not in sys.path:
            sys.path.insert(0, tau_bench_root)
        
        # Storage for loaded data and tool classes
        self.data = {}
        self.tool_classes = {}
        
        # Update env_name for clarity
        self.env_name = f"TauBench_{self.domain.capitalize()}"
    
    def connect(self) -> bool:
        """Load data files (users, products, etc.)"""
        try:
            # Load all JSON data files
            data_files = list(Path(self.data_path).glob("*.json"))
            
            for data_file in data_files:
                data_name = data_file.stem  # e.g., "users", "flights", "reservations"
                
                with open(data_file, 'r') as f:
                    self.data[data_name] = json.load(f)
                
                print(f"   • Loaded {data_name}: {len(self.data[data_name])} records")
            
            print(f"✅ Connected to Tau-bench {self.domain} domain")
            print(f"   • Data loaded: {list(self.data.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Tau-bench: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tools(self) -> List[ToolWrapper]:
        """Load all tool classes from the tools directory"""
        if not self.data:
            raise RuntimeError("Must call connect() first to load data!")
        
        try:
            # Get all .py files in tools directory (except __init__.py)
            tool_files = [
                f for f in Path(self.tools_path).glob("*.py") 
                if f.name != "__init__.py"
            ]
            
            print(f"\n🔧 Loading tools from {self.tools_path}:")
            
            for tool_file in tool_files:
                tool_name = tool_file.stem  # e.g., "book_reservation"
                
                # Load the tool class
                tool_class = self._load_tool_class(tool_file, tool_name)
                
                if tool_class:
                    # Wrap it as a ToolWrapper
                    wrapped_tool = self._wrap_tau_tool(tool_class, tool_name)
                    if wrapped_tool:
                        self.tools.append(wrapped_tool)
                        print(f"   • Loaded: {tool_name}")
            
            print(f"✅ Loaded {len(self.tools)} Tau-bench {self.domain} tools")
            return self.tools
            
        except Exception as e:
            print(f"❌ Failed to load Tau-bench tools: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _load_tool_class(self, tool_file: Path, tool_name: str) -> Optional[type]:
        """
        Dynamically load a tool class from a Python file
        
        Tau-bench tools follow pattern:
        - File: book_reservation.py
        - Class: BookReservation (CamelCase)
        - Methods: invoke() and get_info()
        """
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Convert snake_case to CamelCase
            # e.g., "book_reservation" -> "BookReservation"
            class_name = ''.join(word.capitalize() for word in tool_name.split('_'))
            
            # Try to find the tool class
            if hasattr(module, class_name):
                tool_class = getattr(module, class_name)
                
                # Verify it has required methods
                if hasattr(tool_class, 'invoke') and hasattr(tool_class, 'get_info'):
                    return tool_class
            
            # Fallback: find any class with invoke and get_info methods
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, 'invoke') and hasattr(attr, 'get_info'):
                    return attr
            
            print(f"⚠️  No valid tool class found in {tool_name}")
            return None
            
        except Exception as e:
            print(f"⚠️  Failed to load {tool_name}: {e}")
            return None
    
    def _wrap_tau_tool(self, tool_class: type, tool_name: str) -> Optional[ToolWrapper]:
        """
        Wrap tau-bench tool class into ToolWrapper
        Uses get_info() for schema extraction
        """
        try:
            # Get tool info schema
            info = tool_class.get_info()
            function_info = info.get("function", {})
            
            # Create executor that calls tool_class.invoke with data
            def tool_executor(**kwargs) -> Any:
                """
                Execute tau-bench tool with data injection
                """
                try:
                    # Tau-bench tools expect data as first argument
                    result = tool_class.invoke(self.data, **kwargs)
                    
                    # Try to parse JSON string results
                    if isinstance(result, str):
                        try:
                            return json.loads(result)
                        except:
                            return result
                    return result
                    
                except Exception as e:
                    return {"error": str(e), "success": False}
            
            # Extract arg_schema from get_info()
            parameters = function_info.get("parameters", {})
            arg_schema = {
                "type": parameters.get("type", "object"),
                "properties": parameters.get("properties", {}),
                "required": parameters.get("required", [])
            }
            
            # Infer return schema (tau-bench doesn't provide this in get_info)
            # We'll create a reasonable default based on the tool
            return_schema = self._infer_return_schema(tool_name, function_info)
            
            # Create ToolWrapper
            tool = ToolWrapper(
                name=f"tau_{self.domain}_{tool_name}",
                description=function_info.get("description", ""),
                invoke_fn=tool_executor,
                arg_schema=arg_schema,
                return_schema=return_schema,
                metadata={
                    "env": "tau_bench",
                    "domain": self.domain,
                    "original_name": tool_name,
                    "data_access": list(self.data.keys())
                }
            )
            
            return tool
            
        except Exception as e:
            print(f"⚠️  Failed to wrap tool {tool_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _infer_return_schema(self, tool_name: str, function_info: Dict) -> Dict[str, Any]:
        """
        Infer return schema for tau-bench tools
        Since tau-bench doesn't provide return info in get_info(), we infer from tool name
        """
        description = function_info.get("description", "")
        
        # Pattern matching for common tool types
        if "book" in tool_name or "create" in tool_name:
            return {
                "type": "object",
                "description": f"Created {tool_name.replace('_', ' ')} object with details",
                "properties": {}
            }
        elif "get" in tool_name or "search" in tool_name or "list" in tool_name:
            return {
                "type": "array",
                "description": f"List of {tool_name.replace('_', ' ')} results",
                "items": {
                    "type": "object"
                }
            }
        elif "update" in tool_name or "modify" in tool_name:
            return {
                "type": "object",
                "description": f"Updated {tool_name.replace('_', ' ')} object",
                "properties": {}
            }
        elif "cancel" in tool_name or "delete" in tool_name:
            return {
                "type": "string",
                "description": "Confirmation message for the operation"
            }
        else:
            return {
                "type": "string",
                "description": description or "Tool execution result"
            }
    
    def disconnect(self):
        """Cleanup"""
        self.data = {}
        self.tool_classes = {}
        print(f"🔌 Disconnected from Tau-bench {self.domain}")


class TauBenchRetailAdapter(TauBenchAdapter):
    """Convenience adapter specifically for retail domain"""
    
    def __init__(self, config: Dict[str, Any]):
        config["domain"] = "retail"
        super().__init__(config)


class TauBenchAirlineAdapter(TauBenchAdapter):
    """Convenience adapter specifically for airline domain"""
    
    def __init__(self, config: Dict[str, Any]):
        config["domain"] = "airline"
        super().__init__(config)
