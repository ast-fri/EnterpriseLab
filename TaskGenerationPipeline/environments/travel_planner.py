"""
TravelPlanner environment adapter with ToolWrapper
"""
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from .base import EnvironmentAdapter
from .tool_wrapper import ToolWrapper


class TravelPlannerAdapter(EnvironmentAdapter):
    """
    Adapter for TravelPlanner benchmark
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.travel_planner_path = config.get(
            "travel_planner_path", 
            "./environments/TravelPlanner"
        )
        self.load_planner = config.get("load_planner", False)
        
        self.base_path = Path(self.travel_planner_path)
        self.tools_path = self.base_path / "tools"
        self.database_path = config.get("database_path") or (self.base_path / "database")
        
        self.tool_apis = {}
        
        self.env_name = "TravelPlanner"
    
    def connect(self) -> bool:
        """Load all TravelPlanner tool APIs"""
        try:
            original_cwd = os.getcwd()
            original_path = sys.path.copy()
            
            try:
                os.chdir(str(self.base_path))
                sys.path.insert(0, str(self.base_path))
                
                # Import core tools
                from environments.TravelPlanner.tools.flights.apis import Flights
                from environments.TravelPlanner.tools.accommodations.apis import Accommodations
                from environments.TravelPlanner.tools.restaurants.apis import Restaurants
                from environments.TravelPlanner.tools.attractions.apis import Attractions
                from environments.TravelPlanner.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
                from environments.TravelPlanner.tools.cities.apis import Cities
                from environments.TravelPlanner.tools.notebook.apis import Notebook
                
                # Initialize core tools
                print(f"🔧 Initializing TravelPlanner tools...")
                
                self.tool_apis['flights'] = Flights()
                self.tool_apis['flights'].load_db()
                print(f"   • Flights API loaded")
                
                self.tool_apis['accommodations'] = Accommodations()
                print(f"   • Accommodations API loaded")
                
                self.tool_apis['restaurants'] = Restaurants()
                print(f"   • Restaurants API loaded")
                
                self.tool_apis['attractions'] = Attractions()
                print(f"   • Attractions API loaded")
                
                self.tool_apis['distance_matrix'] = GoogleDistanceMatrix()
                print(f"   • GoogleDistanceMatrix API loaded")
                
                self.tool_apis['cities'] = Cities()
                print(f"   • Cities API loaded")
                
                self.tool_apis['notebook'] = Notebook()
                print(f"   • Notebook API loaded")
                
                # Try to load Planner (optional)
                if self.load_planner:
                    try:
                        from environments.TravelPlanner.tools.planner.apis import Planner
                        self.tool_apis['planner'] = Planner()
                        print(f"   • Planner API loaded")
                    except ImportError as e:
                        print(f"   ⚠️  Planner skipped: {e}")
                
                print(f"✅ Connected to TravelPlanner")
                return True
                
            finally:
                os.chdir(original_cwd)
                sys.path = original_path
            
        except Exception as e:
            print(f"❌ Failed to import TravelPlanner tools: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                os.chdir(original_cwd)
                sys.path = original_path
            except:
                pass
            
            return False
    
    def load_tools(self) -> List[ToolWrapper]:
        """Create ToolWrappers from TravelPlanner APIs"""
        if not self.tool_apis:
            raise RuntimeError("Must call connect() first!")
        
        try:
            # Define tool mappings - FIXED: use string keys, not instances
            tool_definitions = [
                {
                    "name": "flight_search",
                    "api_key": "flights",  # String key, not instance
                    "method": "run",
                    "description": "Search for flights between two cities on a specific date. Args: origin_city, destination_city, date (YYYY-MM-DD). Returns flight information.",
                    "args": ["origin", "destination", "date"]
                },
                {
                    "name": "accommodation_search",
                    "api_key": "accommodations",
                    "method": "run",
                    "description": "Search for accommodations in a city. Args: city. Returns hotel information.",
                    "args": ["city"]
                },
                {
                    "name": "restaurant_search",
                    "api_key": "restaurants",
                    "method": "run",
                    "description": "Search for restaurants in a city. Args: city. Returns restaurant information.",
                    "args": ["city"]
                },
                {
                    "name": "attraction_search",
                    "api_key": "attractions",
                    "method": "run",
                    "description": "Search for attractions in a city. Args: city. Returns attraction information.",
                    "args": ["city"]
                },
                {
                    "name": "distance_matrix",
                    "api_key": "distance_matrix",
                    "method": "run",
                    "description": "Get distance and duration between two cities. Args: origin_city, destination_city. Returns distance and driving info.",
                    "args": ["origin", "destination"]
                },
                {
                    "name": "city_search",
                    "api_key": "cities",
                    "method": "run",
                    "description": "Get information about a city. Args: city_name. Returns city information.",
                    "args": ["city"]
                },
                {
                    "name": "notebook_write",
                    "api_key": "notebook",
                    "method": "write",
                    "description": "Write information to travel planning notebook. Args: content. Records decisions during planning.",
                    "args": ["content"]
                },
            ]
            
            # Add planner if loaded
            if 'planner' in self.tool_apis:
                tool_definitions.append({
                    "name": "planner",
                    "api_key": "planner",
                    "method": "run",
                    "description": "Create comprehensive travel plan. Takes full query with constraints and generates itinerary.",
                    "args": ["query"]
                })
            
            # Create ToolWrappers
            for tool_def in tool_definitions:
                api_key = tool_def["api_key"]
                if api_key not in self.tool_apis:
                    continue
                
                tool = self._create_tool_from_definition(tool_def)
                if tool:
                    self.tools.append(tool)
                    print(f"   • Loaded: {tool_def['name']}")
            
            print(f"✅ Loaded {len(self.tools)} TravelPlanner tools")
            return self.tools
            
        except Exception as e:
            print(f"❌ Failed to load TravelPlanner tools: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_tool_from_definition(self, tool_def: Dict[str, Any]) -> Optional[ToolWrapper]:
        """Create ToolWrapper from tool definition"""
        try:
            # Get API instance using string key
            api_key = tool_def["api_key"]
            api_instance = self.tool_apis[api_key]
            
            method_name = tool_def["method"]
            method = getattr(api_instance, method_name)
            
            # Create executor
            def make_executor(api_method, arg_names):
                def executor(**kwargs) -> Any:
                    try:
                        args = [kwargs.get(arg_name) for arg_name in arg_names]
                        result = api_method(*args)
                        
                        if hasattr(result, 'to_dict'):
                            return result.to_dict()
                        elif hasattr(result, 'to_json'):
                            return result.to_json()
                        else:
                            return result
                    except Exception as e:
                        return {"error": str(e), "success": False}
                
                return executor
            
            # Simple format
            arg_schema = tool_def.get("args", [])
            return_schema = {"result": f"Result from {tool_def['name']}"}
            
            tool = ToolWrapper(
                name=f"travel_{tool_def['name']}",
                description=tool_def["description"],
                invoke_fn=make_executor(method, arg_schema),
                arg_schema=arg_schema,
                return_schema=return_schema,
                metadata={
                    "env": "travel_planner",
                    "original_name": tool_def["name"],
                    "api_class": api_instance.__class__.__name__
                }
            )
            
            return tool
            
        except Exception as e:
            print(f"⚠️  Failed to create tool {tool_def.get('name', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def disconnect(self):
        """Cleanup"""
        self.tool_apis = {}
        print(f"🔌 Disconnected from TravelPlanner")
