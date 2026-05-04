"""
CRMArena environment adapter with ToolWrapper
"""
import os
import sys
from typing import List, Dict, Any, Callable
from pathlib import Path
from dotenv import load_dotenv

from .base import EnvironmentAdapter
from .tool_wrapper import ToolWrapper


class CRMArenaAdapter(EnvironmentAdapter):
    """
    Adapter for CRMArena with live Salesforce API
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                "org_type": "original" | "b2b" | "b2c",
                "crm_arena_path": "./environments/CRMArena",
                "env_vars_loaded": True
            }
        """
        super().__init__(config)
        
        self.org_type = config.get("org_type", "original")
        self.crm_path = config.get("crm_arena_path", "./environments/CRMArena")
        self.sf_connector = None
        self.raw_functions = []
        
        if self.crm_path not in sys.path:
            sys.path.insert(0, self.crm_path)
        
        if not config.get("env_vars_loaded", False):
            load_dotenv()
    
    def connect(self) -> bool:
        """Establish Salesforce connection"""
        try:
            from crm_sandbox.env.connect_sandbox import SalesforceConnector
            
            self.sf_connector = SalesforceConnector(org_type=self.org_type)
            print(f"✅ Connected to CRMArena ({self.org_type}) Salesforce org")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to CRMArena: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tools(self) -> List[ToolWrapper]:
        """Load all tools from CRMArena functions.py"""
        if not self.sf_connector:
            raise RuntimeError("Must call connect() first!")
        
        try:
            from crm_sandbox.env import functions
            
            self.raw_functions = self._discover_crm_functions(functions)
            
            for func in self.raw_functions:
                tool = self._wrap_crm_function(func)
                if tool:
                    self.tools.append(tool)
                    print(f"   • Loaded: {tool.name}")
            
            print(f"✅ Loaded {len(self.tools)} CRMArena tools")
            return self.tools
            
        except Exception as e:
            print(f"❌ Failed to load CRMArena tools: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _discover_crm_functions(self, functions_module) -> List[Callable]:
        """Discover all tool functions from functions.py"""
        discovered = []
        
        for attr_name in dir(functions_module):
            attr = getattr(functions_module, attr_name)
            
            if callable(attr) and hasattr(attr, '__info__'):
                discovered.append(attr)
        
        return discovered
    
    def _wrap_crm_function(self, crm_func: Callable) -> ToolWrapper:
        """
        Wrap CRMArena function into ToolWrapper
        Uses __info__ metadata for schema
        """
        info = crm_func.__info__
        func_name = crm_func.__name__
        
        # Extract schema from __info__
        function_info = info.get("function", {})
        
        # Create executor that injects sf_connector
        def tool_executor(**kwargs) -> Any:
            """Execute with sf_connector injected"""
            try:
                result = crm_func(**kwargs, sf_connector=self.sf_connector)
                return result
            except Exception as e:
                return {"error": str(e), "success": False}
        
        # Extract args_schema from __info__
        parameters = function_info.get("parameters", {})
        args_schema = {
            "type": parameters.get("type", "object"),
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", [])
        }
        
        # Extract FULL return schema from __info__
        returns = function_info.get("returns", {})
        return_schema = {
            "type": returns.get("type", "string"),
            "description": returns.get("description", ""),
            "properties": returns.get("properties", {}),
            "items": returns.get("items", {})  # For array types
        }
        
        # Create ToolWrapper
        tool = ToolWrapper(
            name=f"crm_{self.org_type}_{func_name}",
            description=function_info.get("description", ""),
            invoke_fn=tool_executor,
            args_schema=args_schema,
            return_schema=return_schema,
            metadata={
                "env": "crm_arena",
                "org_type": self.org_type,
                "original_name": func_name,
                "crm_info": info
            }
        )
        
        return tool
    
    def disconnect(self):
        """Cleanup connection"""
        self.sf_connector = None
        print(f"🔌 Disconnected from CRMArena ({self.org_type})")


class CRMArenaLocalAdapter(EnvironmentAdapter):
    """
    Adapter for CRMArena using local SQLite database
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.db_path = config.get("db_path")
        self.db_type = config.get("db_type", "original")
        self.db_conn = None
    
    def connect(self) -> bool:
        """Connect to local SQLite database"""
        import sqlite3
        
        try:
            self.db_conn = sqlite3.connect(self.db_path)
            self.db_conn.row_factory = sqlite3.Row
            print(f"✅ Connected to local CRMArena database: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False
    
    def load_tools(self) -> List[ToolWrapper]:
        """Create tools that query local database"""
        if not self.db_conn:
            raise RuntimeError("Must call connect() first!")
        
        tables = self._get_tables()
        
        for table in tables:
            # Create query function for this table
            def make_query_func(tbl):
                def query_func(**kwargs) -> Any:
                    return self._query_table(tbl, kwargs)
                return query_func
            
            tool = ToolWrapper(
                name=f"crm_local_{self.db_type}_query_{table}",
                description=f"Query {table} table from local CRMArena database",
                invoke_fn=make_query_func(table),
                args_schema={
                    "type": "object",
                    "properties": {
                        "conditions": {
                            "type": "object",
                            "description": "Query conditions as key-value pairs"
                        }
                    },
                    "required": []
                },
                return_schema={
                    "type": "array",
                    "description": f"Array of records from {table} table",
                    "items": {
                        "type": "object",
                        "description": f"A record from {table} table"
                    }
                },
                metadata={
                    "env": "crm_arena_local",
                    "db_type": self.db_type,
                    "table": table,
                    "operation": "query"
                }
            )
            
            self.tools.append(tool)
        
        print(f"✅ Created {len(self.tools)} local database tools")
        return self.tools
    
    def _get_tables(self) -> List[str]:
        """Get list of tables in database"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]
    
    def _query_table(self, table: str, conditions: Dict[str, Any]) -> List[Dict]:
        """Query table with conditions"""
        cursor = self.db_conn.cursor()
        
        where_parts = [f"{k}=?" for k in conditions.keys()]
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        query = f"SELECT * FROM {table} WHERE {where_clause} LIMIT 10"
        
        try:
            cursor.execute(query, tuple(conditions.values()))
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def disconnect(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
        print(f"🔌 Disconnected from CRMArena local database")
