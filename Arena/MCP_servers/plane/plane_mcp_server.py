#!/usr/bin/env python3
"""
Plane MCP Server - HTTP Version
Provides tools to interact with Plane project management via HTTP
"""

import os
import json
import asyncio
import httpx
import logging
from typing import Any
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plane-mcp")

class PlaneAPIClient:
    def __init__(self, api_key: str, workspace_slug: str, host_url: str):
        self.api_key = api_key
        self.workspace_slug = workspace_slug
        self.host_url = host_url.rstrip('/')
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    async def request(self, method: str, endpoint: str, data: dict = None):
        url = f"{self.host_url}/api/v1/workspaces/{self.workspace_slug}{endpoint}"
        logger.info(f"Making {method} request to: {url}")
        if data:
            logger.info(f"Request data: {json.dumps(data)}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                if method == "GET":
                    response = await client.get(url, headers=self.headers)
                elif method == "POST":
                    response = await client.post(url, headers=self.headers, json=data)
                elif method == "PATCH":
                    response = await client.patch(url, headers=self.headers, json=data)
                elif method == "DELETE":
                    response = await client.delete(url, headers=self.headers)
                
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response body: {response.text[:500]}")  # Log first 500 chars
                
                response.raise_for_status()
                logger.info(f"Request successful: {response.status_code}")
                return response.json() if response.text else {}
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code}: {e.response.text}")
                raise Exception(f"API Error {e.response.status_code}: {e.response.text}")
            except httpx.HTTPError as e:
                logger.error(f"API request failed: {str(e)}")
                raise Exception(f"API request failed: {str(e)}")

# Initialize Plane client
PLANE_API_KEY = os.getenv("PLANE_API_KEY","plane_api_769c05cb19ce42008423b71ad72c51d7")
PLANE_WORKSPACE_SLUG = os.getenv("PLANE_WORKSPACE_SLUG","mycompany")
PLANE_API_HOST_URL = os.getenv("PLANE_API_HOST_URL", "http://172.17.0.1:3001")

if not PLANE_API_KEY or not PLANE_WORKSPACE_SLUG:
    raise ValueError("PLANE_API_KEY and PLANE_WORKSPACE_SLUG must be set")

logger.info(f"Initializing Plane MCP Server")
logger.info(f"API Host: {PLANE_API_HOST_URL}")
logger.info(f"Workspace: {PLANE_WORKSPACE_SLUG}")

plane_client = PlaneAPIClient(PLANE_API_KEY, PLANE_WORKSPACE_SLUG, PLANE_API_HOST_URL)

# Create the FastMCP server
mcp = FastMCP("Plane MCP Server")

# User tools
@mcp.tool()
async def get_user() -> str:
    """Get the current user's information"""
    try:
        result = await plane_client.request("GET", "/users/me/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_workspace_members() -> str:
    """Get all members of the workspace"""
    try:
        result = await plane_client.request("GET", "/workspace-members/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Project tools
@mcp.tool()
async def get_projects() -> str:
    """Get all projects for the current user"""
    try:
        result = await plane_client.request("GET", "/projects/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_project(name: str, identifier: str = None) -> str:
    """Create a new project
    
    Args:
        name: Project name
        identifier: Project identifier (optional, will be auto-generated from name if not provided)
    """
    try:
        # Generate identifier from name if not provided
        if not identifier:
            identifier = name.upper().replace(" ", "")[:5]
        
        data = {
            "name": name,
            "identifier": identifier
        }
        result = await plane_client.request("POST", "/projects/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Issue Type tools
@mcp.tool()
async def list_issue_types(project_id: str) -> str:
    """Get all issue types for a specific project"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/issue-types/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_issue_type(project_id: str, type_id: str) -> str:
    """Get details of a specific issue type"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/issue-types/{type_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_issue_type(project_id: str, name: str, description: str) -> str:
    """Create a new issue type in a project"""
    try:
        issue_type_data = {"name": name, "description": description}
        result = await plane_client.request("POST", f"/projects/{project_id}/issue-types/", issue_type_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_issue_type(project_id: str, type_id: str, issue_type_data: str) -> str:
    """Update an existing issue type"""
    try:
        data = json.loads(issue_type_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/issue-types/{type_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_issue_type(project_id: str, type_id: str) -> str:
    """Delete an issue type"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/issue-types/{type_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# State tools
@mcp.tool()
async def list_states(project_id: str) -> str:
    """Get all states for a specific project"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/states/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_state(project_id: str, state_id: str) -> str:
    """Get details of a specific state"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/states/{state_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_state(project_id: str, name: str, color: str) -> str:
    """Create a new state in a project"""
    try:
        state_data = {"name": name, "color": color}
        result = await plane_client.request("POST", f"/projects/{project_id}/states/", state_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_state(project_id: str, state_id: str, state_data: str) -> str:
    """Update an existing state"""
    try:
        data = json.loads(state_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/states/{state_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_state(project_id: str, state_id: str) -> str:
    """Delete a state"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/states/{state_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Label tools
@mcp.tool()
async def list_labels(project_id: str) -> str:
    """Get all labels for a specific project"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/labels/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_label(project_id: str, label_id: str) -> str:
    """Get details of a specific label"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/labels/{label_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_label(project_id: str, name: str, color: str) -> str:
    """Create a new label in a project"""
    try:
        label_data = {"name": name, "color": color}
        result = await plane_client.request("POST", f"/projects/{project_id}/labels/", label_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_label(project_id: str, label_id: str, label_data: str) -> str:
    """Update an existing label"""
    try:
        data = json.loads(label_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/labels/{label_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_label(project_id: str, label_id: str) -> str:
    """Delete a label"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/labels/{label_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Issue tools
@mcp.tool()
async def get_issue_using_readable_identifier(project_identifier: str, issue_identifier: str) -> str:
    """Get issue details using readable identifier (e.g., PROJ-123)"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_identifier}/issues/{issue_identifier}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_issue_comments(project_id: str, issue_id: str) -> str:
    """Get all comments for a specific issue"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/issues/{issue_id}/comments/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def add_issue_comment(project_id: str, issue_id: str, comment_html: str) -> str:
    """Add a comment to an issue"""
    try:
        result = await plane_client.request("POST", f"/projects/{project_id}/issues/{issue_id}/comments/", {"comment_html": comment_html})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def plane_create_issue(project_id: str, name: str, description_html: str) -> str:
    """Create a new issue in a plane project"""
    try:
        issue_data = {"name": name, "description_html": description_html}
        result = await plane_client.request("POST", f"/projects/{project_id}/issues/", issue_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_cycle_issue(project_id: str, cycle_id: str, name: str, description_html: str) -> str:
    """Create a new issue directly in a cycle"""
    try:
        issue_data = {"name": name, "description_html": description_html, "project_id": project_id}
        issue = await plane_client.request("POST", f"/projects/{project_id}/issues/", issue_data)
        await plane_client.request("POST", f"/projects/{project_id}/cycles/{cycle_id}/cycle-issues/", {"issues": [issue['id']]})
        return json.dumps(issue, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_issue(project_id: str, issue_id: str, issue_data: str) -> str:
    """Update an existing issue"""
    try:
        data = json.loads(issue_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/issues/{issue_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_issue(project_id: str, issue_id: str) -> str:
    """Delete an issue"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/issues/{issue_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Module tools
@mcp.tool()
async def list_modules(project_id: str) -> str:
    """Get all modules for a specific project"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/modules/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_module(project_id: str, module_id: str) -> str:
    """Get details of a specific module"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/modules/{module_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_module(project_id: str, name: str) -> str:
    """Create a new module in a project"""
    try:
        module_data = {"name": name}
        result = await plane_client.request("POST", f"/projects/{project_id}/modules/", module_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_module(project_id: str, module_id: str, module_data: str) -> str:
    """Update an existing module"""
    try:
        data = json.loads(module_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/modules/{module_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_module(project_id: str, module_id: str) -> str:
    """Delete a module"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/modules/{module_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def list_module_issues(project_id: str, module_id: str) -> str:
    """Get all issues for a specific module"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/modules/{module_id}/module-issues/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def add_module_issues(project_id: str, module_id: str, issues: str) -> str:
    """Add issues to a module"""
    try:
        issue_list = json.loads(issues)
        result = await plane_client.request("POST", f"/projects/{project_id}/modules/{module_id}/module-issues/", {"issues": issue_list})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_module_issue(project_id: str, module_id: str, issue_id: str) -> str:
    """Remove an issue from a module"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/modules/{module_id}/module-issues/{issue_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Cycle tools
@mcp.tool()
async def list_cycles(project_id: str) -> str:
    """Get all cycles for a specific project"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/cycles/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_cycle(project_id: str, cycle_id: str) -> str:
    """Get details of a specific cycle"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/cycles/{cycle_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_cycle(project_id: str, name: str, start_date: str, end_date: str) -> str:
    """Create a new cycle in a project"""
    try:
        cycle_data = {"name": name, "start_date": start_date, "end_date": end_date, "project_id": project_id}
        result = await plane_client.request("POST", f"/projects/{project_id}/cycles/", cycle_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_cycle(project_id: str, cycle_id: str, cycle_data: str) -> str:
    """Update an existing cycle"""
    try:
        data = json.loads(cycle_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/cycles/{cycle_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_cycle(project_id: str, cycle_id: str) -> str:
    """Delete a cycle"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/cycles/{cycle_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def transfer_cycle_issues(project_id: str, cycle_id: str, new_cycle_id: str) -> str:
    """Transfer issues from one cycle to another"""
    try:
        result = await plane_client.request("POST", f"/projects/{project_id}/cycles/{cycle_id}/transfer-issues/", {"new_cycle_id": new_cycle_id})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def list_cycle_issues(project_id: str, cycle_id: str) -> str:
    """Get all issues for a specific cycle"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/cycles/{cycle_id}/cycle-issues/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def add_cycle_issues(project_id: str, cycle_id: str, issues: str) -> str:
    """Add existing issues to a cycle"""
    try:
        issue_list = json.loads(issues)
        result = await plane_client.request("POST", f"/projects/{project_id}/cycles/{cycle_id}/cycle-issues/", {"issues": issue_list})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_cycle_issue(project_id: str, cycle_id: str, issue_id: str) -> str:
    """Remove an issue from a cycle"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/cycles/{cycle_id}/cycle-issues/{issue_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Worklog tools
@mcp.tool()
async def get_issue_worklogs(project_id: str, issue_id: str) -> str:
    """Get all worklogs for a specific issue"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/issues/{issue_id}/worklogs/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_total_worklogs(project_id: str) -> str:
    """Get total logged time for a project"""
    try:
        result = await plane_client.request("GET", f"/projects/{project_id}/worklogs/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_worklog(project_id: str, issue_id: str, description: str, duration: int) -> str:
    """Create a new worklog for an issue"""
    try:
        worklog_data = {"description": description, "duration": duration}
        result = await plane_client.request("POST", f"/projects/{project_id}/issues/{issue_id}/worklogs/", worklog_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def update_worklog(project_id: str, issue_id: str, worklog_id: str, worklog_data: str) -> str:
    """Update an existing worklog"""
    try:
        data = json.loads(worklog_data)
        result = await plane_client.request("PATCH", f"/projects/{project_id}/issues/{issue_id}/worklogs/{worklog_id}/", data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_worklog(project_id: str, issue_id: str, worklog_id: str) -> str:
    """Delete a worklog"""
    try:
        result = await plane_client.request("DELETE", f"/projects/{project_id}/issues/{issue_id}/worklogs/{worklog_id}/")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=12023,
        )
    )