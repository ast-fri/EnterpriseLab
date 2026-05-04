# utils.py
"""
Shared utility functions
"""




"""
Shared utility functions for EnterpriseArena-Genesis
Updated to use AzureChatOpenAI with JSON response support
"""

import json
import hashlib
import time
import os
from typing import Any, Dict, Callable
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from AutoQuest.intelligent_explorer.tool_classifier import ToolOperation


class GPTCaller:
    """
    Wrapper for GPT API calls using AzureChatOpenAI
    Supports both JSON mode and text responses
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        api_version: str = "2024-10-21",
        model_name: str = "gpt-4o",
        max_retries: int = 5
    ):
        """
        Initialize GPT caller with Azure configuration
        
        Args:
            api_key: Azure API key (if None, reads from AZURE_CHAT_API_KEY env var)
            api_base: Azure endpoint (if None, reads from AZURE_CHAT_ENDPOINT env var)
            api_version: Azure API version
            model_name: Model deployment name
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key or os.getenv("AZURE_CHAT_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_CHAT_ENDPOINT")
        self.api_version = api_version
        self.model_name = model_name
        self.max_retries = max_retries
        
        if not self.api_key or not self.api_base:
            raise ValueError(
                "Azure API key and endpoint must be provided either as arguments "
                "or via AZURE_CHAT_API_KEY and AZURE_CHAT_ENDPOINT environment variables"
            )
        
        # Initialize base LLM (without JSON mode)
        self.llm = AzureChatOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            model_name=self.model_name,
            temperature=0.3  # Default, can be overridden per call
        )
        
        # Initialize JSON mode LLM
        self.llm_json = self.llm.bind(
            response_format={"type": "json_object"}
        )
    
    async def __call__(
        self,
        prompt: str,
        response_format: str = "json",
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 16384
    ) -> Dict[str, Any]:
        """
        Call GPT with prompt and return response
        
        Args:
            prompt: The prompt to send
            response_format: "json" or "text"
            model: Model to use (currently ignored, uses self.model_name)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Parsed JSON response if response_format="json", else dict with "response" key
        """
        
        # Create LLM with specified temperature
        if response_format == "json":
            llm = self.llm.bind(
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Add system message to ensure JSON output
            system_msg = "You are a helpful AI assistant. Always respond in valid JSON format."
        else:
            llm = AzureChatOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                model_name=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            system_msg = "You are a helpful AI assistant."
        
        # Retry logic
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Create messages
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]
                
                # Call the model
                response = llm.invoke(messages)
                
                # Extract content
                content = response.content
                
                # Parse based on response format
                if response_format == "json":
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON response: {e}")
                        print(f"Raw content: {content[:200]}...")
                        # Retry if JSON parsing fails
                        raise e
                else:
                    return {"response": content}
            
            except Exception as e:
                last_error = e
                retries += 1
                time_wait = 15 * retries
                
                print(f"❌ GPT call failed (attempt {retries}/{self.max_retries})")
                print(f"   Error: {str(e)}")
                print(f"   Prompt length: {len(prompt)} chars")
                
                if retries < self.max_retries:
                    print(f"   Retrying in {time_wait} seconds...")
                    time.sleep(time_wait)
                else:
                    print(f"   All retries exhausted. Returning empty response.")
        
        # If all retries fail
        print(f"⚠️  All {self.max_retries} retry attempts failed")
        print(f"   Last error: {last_error}")
        
        if response_format == "json":
            return {}  # Empty dict for JSON mode
        else:
            return {"response": "", "error": str(last_error)}
    
    def sync_call(
        self,
        prompt: str,
        response_format: str = "json",
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Synchronous version of call (for non-async contexts)
        
        Args:
            prompt: The prompt to send
            response_format: "json" or "text"
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Parsed JSON response if response_format="json", else dict with "response" key
        """
        
        # Create LLM with specified temperature
        if response_format == "json":
            llm = self.llm.bind(
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )
            system_msg = "You are a helpful AI assistant. Always respond in valid JSON format."
        else:
            llm = AzureChatOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                model_name=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            system_msg = "You are a helpful AI assistant."
        
        # Retry logic
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]
                
                # Synchronous invoke
                response = llm.invoke(messages)
                content = response.content
                
                if response_format == "json":
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON response: {e}")
                        raise e
                else:
                    return {"response": content}
            
            except Exception as e:
                last_error = e
                retries += 1
                time_wait = 15 * retries
                
                print(f"❌ GPT call failed (attempt {retries}/{self.max_retries})")
                print(f"   Error: {str(e)}")
                
                if retries < self.max_retries:
                    print(f"   Retrying in {time_wait} seconds...")
                    time.sleep(time_wait)
        
        # If all retries fail
        if response_format == "json":
            return {}
        else:
            return {"response": "", "error": str(last_error)}


def hash_dict(d: Dict) -> str:
    """Create deterministic hash of dictionary"""
    json_str = json.dumps(d, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def extract_field_names(schema: Dict) -> list:
    """Extract field names from JSON schema"""
    if not schema or not isinstance(schema, dict):
        return []
    
    properties = schema.get("properties", {})
    return list(properties.keys())


async def load_mcp_tools(session: Any) -> list:
    """
    Load tools from MCP session
    
    Args:
        session: MCP session object
    
    Returns:
        List of tools
    """
    tools = []
    try:
        # Assuming session has a list_tools method
        # Adjust based on your actual MCP client API
        tools_response = await session.list_tools()
        
        if hasattr(tools_response, 'tools'):
            tools = tools_response.tools
        elif isinstance(tools_response, list):
            tools = tools_response
        else:
            print(f"Warning: Unexpected tools response format: {type(tools_response)}")
    
    except Exception as e:
        print(f"Error loading tools from session: {e}")
    
    return tools


def save_json(data: Any, filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved to {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_tool_info(tool: Any) -> Dict[str, Any]:
    """
    Format tool information for display/logging
    
    Args:
        tool: Tool object from MCP
    
    Returns:
        Dictionary with formatted tool info
    """
    return {
        "name": getattr(tool, 'name', 'unknown'),
        "description": getattr(tool, 'description', 'No description'),
        "server": getattr(tool, 'server_name', 'unknown'),
        "has_schema": hasattr(tool, 'arg_schema')
    }


def compute_token_estimate(text: str) -> int:
    """
    Rough estimate of token count
    (OpenAI uses ~4 chars per token on average)
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text for display
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def validate_json_response(response: Any) -> bool:
    """
    Validate that response is valid JSON structure
    
    Args:
        response: Response to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    # Check if it's an empty dict or error response
    if not response or "error" in response:
        return False
    
    return True


def create_backup(filepath: str):
    """
    Create backup of file before overwriting
    
    Args:
        filepath: Path to file to backup
    """
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"📦 Created backup: {backup_path}")


# Progress bar utility
class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar"""
        if self.total == 0:
            percentage = 100
        else:
            percentage = (self.current / self.total) * 100
        
        elapsed = time.time() - self.start_time
        
        # Estimate time remaining
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"ETA: {int(remaining)}s"
        else:
            eta_str = "ETA: calculating..."
        
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r  {self.description}: |{bar}| {percentage:.1f}% ({self.current}/{self.total}) {eta_str}", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def complete(self):
        """Mark as complete"""
        self.current = self.total
        self._print_progress()
        


def normalize_operation(operation) -> ToolOperation:
    """
    Normalize operation to enum, handling both string and enum inputs
    
    Args:
        operation: String or ToolOperation enum
        
    Returns:
        ToolOperation enum value
    """
    if isinstance(operation, str):
        return getattr(ToolOperation, operation.upper(), ToolOperation.UNKNOWN)
    elif isinstance(operation, ToolOperation):
        return operation
    return ToolOperation.UNKNOWN
