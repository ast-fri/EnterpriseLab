"""
EnterpriseBench environment adapter
Loads tools from the Tools class which has methods for multiple enterprise domains
"""
import os
import sys
import inspect
from typing import List, Dict, Any, Optional
from pathlib import Path
from .tool_wrapper import ToolWrapper
from dotenv import load_dotenv
import json
from .base import EnvironmentAdapter


class EnterpriseBenchAdapter(EnvironmentAdapter):
    """
    Adapter for EnterpriseBench
    Uses the Tools class which contains methods for:
    - GitHub (repositories, issues)
    - Email (Enterprise Mail System)
    - Collaboration (messages/conversations)
    - CRM (products, customers, sales, support chats, sentiment)
    - IT Service Management (tickets)
    - HR Management (employee records)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                "enterprise_bench_path": "./environments/EnterpriseBench",
                
            }
        """
        super().__init__(config)
        
        self.enterprise_bench_path = config.get(
            "enterprise_bench_path",
            "./environments/EnterpriseBench"
        )
        
        self.base_path = Path(self.enterprise_bench_path)
        
        # Add EnterpriseBench to Python path
        if str(self.base_path) not in sys.path:
            sys.path.insert(0, str(self.base_path))
            
        self.tool_info_path = "/EnterprisePlatform/TaskGenerationPipeline/environments/EnterpriseBench/Task_Generation/utils/tools.json"
        with open(self.tool_info_path, 'r')as f:
            self.tool_info = json.load(f)
        # Tools instance
        self.tools_instance = None
        
        self.env_name = "EnterpriseBench"
    
    def connect(self) -> bool:
        """Initialize EnterpriseBench Tools class"""
        try:
            # Change to EnterpriseBench directory for relative imports
            original_cwd = os.getcwd()
            
            try:
                os.chdir(str(self.base_path))
                
                # Import Tools class
                from environments.EnterpriseBench.tools import Tools
                
                # Instantiate Tools
                self.tools_instance = Tools()
                
                print(f"✅ Connected to EnterpriseBench")
                print(f"   • Tools class instantiated")
                
                return True
                
            finally:
                os.chdir(original_cwd)
            
        except ImportError as e:
            print(f"❌ Failed to import EnterpriseBench Tools: {e}")
            print(f"   Make sure tools.py exists in {self.base_path}")
            import traceback
            traceback.print_exc()
            return False
            
        except Exception as e:
            print(f"❌ Failed to connect to EnterpriseBench: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tools(self) -> List[ToolWrapper]:
        """Load all tool methods from Tools class"""
        if not self.tools_instance:
            raise RuntimeError("Must call connect() first!")
        
        try:
            print(f"\n🔧 Loading EnterpriseBench tools...")
            
            # Get all methods from Tools instance
            tool_methods = self._discover_tool_methods()
            
            # Convert each method to a ToolWrapper
            for method_name, method in tool_methods.items():
                tool = self._wrap_tool_method(method_name, method)
                if tool:
                    self.tools.append(tool)
                    print(f"   • Loaded: {method_name}")
            
            print(f"✅ Loaded {len(self.tools)} EnterpriseBench tools")
            return self.tools
            
        except Exception as e:
            print(f"❌ Failed to load EnterpriseBench tools: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _discover_tool_methods(self) -> Dict[str, callable]:
        """
        Discover all tool methods from Tools class
        Excludes internal methods and helpers
        """
        tool_methods = {}
        
        # Get all methods from Tools instance
        for name, method in inspect.getmembers(self.tools_instance, predicate=inspect.ismethod):
            # Skip private/internal methods
            if name.startswith('_'):
                continue
            
            # Skip utility methods
            if name in ['get_tool_context', 'load_json', 'save_json', 'get_emp_id_by_email', 'llm']:
                continue
            
            # This is a tool method
            tool_methods[name] = method
        
        return tool_methods
    
    def _wrap_tool_method(self, method_name: str, method: callable) -> Optional[ToolWrapper]:
        """
        Wrap a Tools method into a ToolWrapper
        """
        try:
            # Extract docstring as description
            tool_instance = next((tool for tool in self.tool_info if method_name == tool["name"]), None) 
            description = tool_instance["description"]
            description = description.strip()
            
            # Create executor that calls the tool method
            def executor(**kwargs) -> str:
                """Execute EnterpriseBench tool"""
                try:
                    # The Tools methods expect an 'arguments' dict
                    result = method(arguments=kwargs)
                    
                    # Convert result to string
                    if isinstance(result, list):
                        if len(result) == 0:
                            return "No results found"
                        return str(result)
                    elif isinstance(result, dict):
                        return str(result)
                    else:
                        return str(result)
                        
                except Exception as e:
                    return f"Error: {str(e)}"
                
            
            # Create ToolWrapper
            tool = ToolWrapper(
                name=f"{tool_instance["name"]}",
                description=tool_instance["description"],
                invoke_fn=executor,
                args_schema=tool_instance["arguments"],
                return_schema=tool_instance["return_type"]
            )
            
            # Add metadata
            tool.metadata = {
                "env": "enterprise_bench",
                "original_name": method_name,
                "domain": self._classify_tool_domain(method_name)
            }
            
            return tool
            
        except Exception as e:
            print(f"⚠️  Failed to wrap tool {method_name}: {e}")
            return None
    
    def _classify_tool_domain(self, method_name: str) -> str:
        """Classify tool into domain based on name prefix"""
        if method_name.startswith('github_'):
            return "github"
        elif method_name.startswith('send_') or method_name.startswith('edit_') or method_name.startswith('delete_message') or method_name.startswith('list_conversation') or method_name.startswith('fetch_conversation'):
            return "collaboration"
        elif 'email' in method_name:
            return "email"
        elif 'customer' in method_name or 'product' in method_name or 'sales' in method_name or 'crm' in method_name or 'sentiment' in method_name:
            return "crm"
        elif 'it_ticket' in method_name or 'ticket' in method_name:
            return "it_service"
        elif 'employee' in method_name:
            return "hr"
        elif 'social_platform' in method_name:
            return "social"
        elif 'overflow' in method_name:
            return "overflow"
        else:
            return "general"
    
    def disconnect(self):
        """Cleanup"""
        self.tools_instance = None
        print(f"🔌 Disconnected from EnterpriseBench")
