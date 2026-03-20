"""
Aider MCP Server - HTTP Version
A Model Context Protocol server for interacting with Aider via HTTP
"""

import json
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastmcp import FastMCP
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
mcp = FastMCP("Aider MCP Server")

@mcp.tool()
async def aider_execute(
    command: str,
    files: List[str] = None,
    working_directory: str = "."
) -> str:
    """
    Execute Aider commands for code editing, file creation, directory listing, git operations, and other development tasks.
    
    Args:
        command: The command to execute with Aider (e.g., 'Create a new Python file', 'Show git status', 'List files in current directory', 'Edit the main function')
        files: List of file paths to include in the context (optional)
        working_directory: Working directory path (defaults to current directory)
    
    Returns:
        Result of the Aider command execution
    """
    if files is None:
        files = []
    
    try:
        # Change to working directory if specified
        original_cwd = os.getcwd()
        if working_directory != ".":
            os.chdir(working_directory)

        # If no files specified, use the working directory
        if not files:
            files = [working_directory]

        # Create model using environment variables
        model_name = os.getenv("AIDER_MODEL", "azure/gpt-4o")
        model = Model(model_name)

        # Create non-interactive IO handler
        io = InputOutput(
            yes=True,
            pretty=False,
            input_history_file=None,
            chat_history_file=None
        )

        # Create coder object with non-interactive settings
        coder = Coder.create(
            main_model=model, 
            fnames=files,
            io=io,
            auto_commits=os.getenv("AIDER_AUTO_COMMITS", "false").lower() == "true",
            dirty_commits=os.getenv("AIDER_DIRTY_COMMITS", "true").lower() == "true",
            dry_run=os.getenv("AIDER_DRY_RUN", "false").lower() == "true"
        )

        # Execute the command
        result = coder.run(command)

        # Restore original working directory
        os.chdir(original_cwd)

        # Return result or confirmation message
        if result:
            return f"Command executed successfully:\n{result}"
        else:
            return f"Command '{command}' executed successfully."

    except Exception as e:
        # Restore original working directory in case of error
        try:
            os.chdir(original_cwd)
        except:
            pass
        return f"Error executing Aider command: {str(e)}"

if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=12011,
        )
    )