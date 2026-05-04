"""
Universal tool wrapper for normalizing tools across all environments
"""
from typing import Any, Dict, Optional, Callable
import inspect


import asyncio
import inspect
from typing import Callable, Optional, Dict, Any

class ToolWrapper:
    """
    Standardized wrapper for tools from any environment
    Provides unified interface: name, description, args_schema, return_schema, invoke
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        invoke_fn: Callable,
        args_schema: Optional[Dict[str, Any]] = None,
        return_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            name: Tool name
            description: Tool description
            invoke_fn: Function to call when tool is invoked (sync or async)
            args_schema: JSON schema for tool arguments
            return_schema: JSON schema for return value (type, description, properties, etc.)
            metadata: Additional metadata (env, domain, etc.)
        """
        self.name = name
        self.description = description
        self._invoke_fn = invoke_fn
        self.args_schema = args_schema or self._infer_arg_schema()
        self.return_schema = return_schema or {"type": "string", "description": "Tool execution result"}
        self.metadata = metadata or {}
        
        # ✅ Detect if the function is async
        self._is_async = asyncio.iscoroutinefunction(invoke_fn)
    def _parse_string_output(self, output: str) -> Any:
        """Try multiple strategies to parse string output"""
        import json
        import ast

        # Strategy 1: JSON parsing (for proper JSON with double quotes)
        try:
            return json.loads(output)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Python literal eval (for single-quoted dicts/lists)
        try:
            return ast.literal_eval(output)
        except (SyntaxError, ValueError):
            pass

        # Strategy 3: Replace single quotes with double quotes and retry JSON
        try:
            # Simple replacement (may not work for all cases)
            fixed = output.replace("'", '"')
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

        # All strategies failed, return None to keep as string
        return None
    async def ainvoke(self, **kwargs) -> Any:
        """
        Async invoke the tool with given arguments
        
        Args:
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        try:
            if self._is_async:
                result = await self._invoke_fn(**kwargs)
            else:
                # Run sync function in executor to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._invoke_fn(**kwargs)
                )
             # ✅ Parse string outputs (JSON or Python syntax)
            if isinstance(result, str):
                # Try multiple parsing strategies
                parsed = self._parse_string_output(result)
                if parsed is not None:
                    return parsed
            return result
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def invoke(self, **kwargs) -> Any:
        """
        Synchronous invoke (for backward compatibility)
        
        Args:
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        try:
            if self._is_async:
                # If function is async, we need to run it in event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - this shouldn't be called
                    raise RuntimeError(
                        f"Tool {self.name} is async but invoke() was called from async context. "
                        f"Use ainvoke() instead."
                    )
                result = loop.run_until_complete(self._invoke_fn(**kwargs))
            else:
                result = self._invoke_fn(**kwargs)
            return result
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _infer_arg_schema(self) -> Dict[str, Any]:
        """Infer argument schema from function signature"""
        try:
            sig = inspect.signature(self._invoke_fn)
            schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls', 'kwargs']:
                    continue
                
                # Infer type from annotation
                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == dict:
                        param_type = "object"
                    elif param.annotation == list:
                        param_type = "array"
                
                schema["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Check if required (no default value)
                if param.default == inspect.Parameter.empty:
                    schema["required"].append(param_name)
            
            return schema
            
        except Exception as e:
            return {"type": "object", "properties": {}, "required": []}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "return_schema": self.return_schema,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        return f"ToolWrapper(name={self.name}, env={self.metadata.get('env', 'unknown')})"
