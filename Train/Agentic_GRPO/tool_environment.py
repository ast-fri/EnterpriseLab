"""
Example Tool Environment for Agentic GRPO.

Demonstrates how to implement a ToolEnvironment with:
- Isolated workspaces per trajectory
- Common tools (search, calculate, read_file)
- Proper error handling
- Timeout protection

CRITICAL: Each trajectory gets its own environment instance to prevent
cross-contamination in parallel execution.
"""

import os
import tempfile
import shutil
import time
from typing import Dict, Any, Protocol

from data_structures import ToolExecutionResult, ToolExecutionStatus


class ToolEnvironment(Protocol):
    """Protocol for tool execution environments."""

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool and return structured result."""
        ...

    def reset(self) -> None:
        """Reset environment state."""
        ...


class SimpleToolEnvironment:
    """
    Simple tool environment with basic utilities.

    Each instance has its own isolated workspace (temp directory).
    """

    def __init__(self, workspace_dir: str = None):
        """
        Initialize environment with isolated workspace.

        Args:
            workspace_dir: Optional workspace directory. If None, creates temp dir.
        """
        # Create isolated workspace
        if workspace_dir is None:
            self.workspace = tempfile.mkdtemp(prefix="agentic_grpo_")
        else:
            self.workspace = workspace_dir
            os.makedirs(self.workspace, exist_ok=True)

        # Tool registry
        self.tools = {
            'search': self._search,
            'calculate': self._calculate,
            'read_file': self._read_file,
            'write_file': self._write_file,
            'list_files': self._list_files
        }

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to execute
            args: Dictionary of arguments

        Returns:
            ToolExecutionResult with status and output
        """
        start_time = time.time()

        # Check if tool exists
        if tool_name not in self.tools:
            return ToolExecutionResult(
                status=ToolExecutionStatus.TOOL_NOT_FOUND,
                output=f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Tool '{tool_name}' not found"
            )

        # Execute tool
        try:
            output = self.tools[tool_name](args)
            return ToolExecutionResult(
                status=ToolExecutionStatus.SUCCESS,
                output=str(output),
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return ToolExecutionResult(
                status=ToolExecutionStatus.RUNTIME_ERROR,
                output=f"Error executing {tool_name}: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

    def reset(self):
        """Clean up workspace."""
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace, ignore_errors=True)

    # ========================================================================
    # Tool Implementations
    # ========================================================================

    def _search(self, args: Dict) -> str:
        """
        Search tool (placeholder - replace with real search API).

        Args:
            query: Search query string
        """
        query = args.get('query', '')

        # TODO: Replace with actual search (SerpAPI, Google Custom Search, etc.)
        # For now, return placeholder
        return f"Search results for '{query}': [Placeholder - implement real search]"

    def _calculate(self, args: Dict) -> str:
        """
        Calculate mathematical expression.

        Args:
            expression: Mathematical expression to evaluate

        WARNING: Using eval() is dangerous in production. Use a safe math parser.
        """
        expression = args.get('expression', '')

        if not expression:
            raise ValueError("Missing 'expression' argument")

        try:
            # SECURITY WARNING: eval() is dangerous!
            # In production, use: sympy, numexpr, or a custom parser
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        except Exception as e:
            raise ValueError(f"Invalid expression '{expression}': {e}")

    def _read_file(self, args: Dict) -> str:
        """
        Read file from workspace.

        Args:
            path: File path (relative to workspace)
        """
        filepath = args.get('path', '')

        if not filepath:
            raise ValueError("Missing 'path' argument")

        # Security: Ensure path is within workspace
        full_path = os.path.join(self.workspace, filepath)
        abs_path = os.path.abspath(full_path)

        if not abs_path.startswith(os.path.abspath(self.workspace)):
            raise PermissionError(f"Path '{filepath}' is outside workspace")

        # Read file
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File '{filepath}' not found")

        with open(abs_path, 'r') as f:
            content = f.read()

        return content

    def _write_file(self, args: Dict) -> str:
        """
        Write file to workspace.

        Args:
            path: File path (relative to workspace)
            content: Content to write
        """
        filepath = args.get('path', '')
        content = args.get('content', '')

        if not filepath:
            raise ValueError("Missing 'path' argument")

        # Security: Ensure path is within workspace
        full_path = os.path.join(self.workspace, filepath)
        abs_path = os.path.abspath(full_path)

        if not abs_path.startswith(os.path.abspath(self.workspace)):
            raise PermissionError(f"Path '{filepath}' is outside workspace")

        # Create directory if needed
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # Write file
        with open(abs_path, 'w') as f:
            f.write(content)

        return f"File '{filepath}' written successfully ({len(content)} bytes)"

    def _list_files(self, args: Dict) -> str:
        """
        List files in workspace directory.

        Args:
            path: Directory path (relative to workspace), default is root
        """
        dirpath = args.get('path', '')

        # Security: Ensure path is within workspace
        full_path = os.path.join(self.workspace, dirpath)
        abs_path = os.path.abspath(full_path)

        if not abs_path.startswith(os.path.abspath(self.workspace)):
            raise PermissionError(f"Path '{dirpath}' is outside workspace")

        # List files
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Directory '{dirpath}' not found")

        files = os.listdir(abs_path)

        if not files:
            return "Directory is empty"

        return "\n".join(files)