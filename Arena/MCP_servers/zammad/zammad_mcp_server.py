#!/usr/bin/env python3
import asyncio
import os
import httpx
from typing import Any
from fastmcp import FastMCP
from enum import Enum

# Initialize FastMCP server
mcp = FastMCP("Zammad MCP Server")

# Zammad configuration
ZAMMAD_URL = os.getenv("ZAMMAD_URL", "http://localhost:8050")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "9ujsvYWntuplYox4uYxXnqDz3rULoMJKxYoRBAYRX5_ixCI2xAExfvVAGVVL5pP5")

def get_headers():
    return {"Authorization": f"Token token={ZAMMAD_TOKEN}"}

class ResponseFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"

class ArticleType(str, Enum):
    NOTE = "note"
    EMAIL = "email"
    PHONE = "phone"
    WEB = "web"

# Tickets
@mcp.tool()
async def zammad_search_tickets(
    query: str = "", 
    state: str | None = None,
    priority: str | None = None,
    group: str | None = None,
    owner: str | None = None,
    customer: str | None = None,
    page: int = 1, 
    per_page: int = 10,
    response_format: ResponseFormat = ResponseFormat.MARKDOWN
) -> dict:
    """Search for tickets with various filters"""
    async with httpx.AsyncClient() as client:
        params = {"page": page, "per_page": per_page}
        if query:
            params["query"] = query
        if state:
            params["state"] = state
        if priority:
            params["priority"] = priority
        if group:
            params["group"] = group
        if owner:
            params["owner"] = owner
        if customer:
            params["customer"] = customer
            
        r = await client.get(f"{ZAMMAD_URL}/api/v1/tickets/search", 
                            headers=get_headers(), params=params)
        return r.json()

@mcp.tool()
async def zammad_get_ticket(
    ticket_id: int,
    include_articles: bool = True,
    article_limit: int = 20,
    article_offset: int = 0
) -> dict:
    """Get ticket details by ID (use internal ID, not display number)"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/tickets/{ticket_id}", 
                            headers=get_headers())
        ticket = r.json()
        
        if include_articles:
            articles_r = await client.get(
                f"{ZAMMAD_URL}/api/v1/ticket_articles/by_ticket/{ticket_id}",
                headers=get_headers(),
                params={"offset": article_offset, "limit": article_limit}
            )
            ticket["articles"] = articles_r.json()
        
        return ticket

@mcp.tool()
async def zammad_create_ticket(
    title: str,
    group: str,
    customer: str,
    article_body: str,
    state: str | None = None,
    priority: str | None = None,
    owner: str | None = None
) -> dict:
    """Create a new ticket"""
    async with httpx.AsyncClient() as client:
        data: dict[str, Any] = {
            "title": title,
            "group": group,
            "customer": customer,
            "article": {
                "subject": title,
                "body": article_body,
                "type": "note",
                "internal": False
            }
        }
        if state:
            data["state"] = state
        if priority:
            data["priority"] = priority
        if owner:
            data["owner"] = owner
            
        r = await client.post(f"{ZAMMAD_URL}/api/v1/tickets", 
                             headers=get_headers(), json=data)
        return r.json()

@mcp.tool()
async def zammad_update_ticket(
    ticket_id: int,
    title: str | None = None,
    state: str | None = None,
    priority: str | None = None,
    owner: str | None = None,
    group: str | None = None
) -> dict:
    """Update ticket (use internal ID, not display number)"""
    async with httpx.AsyncClient() as client:
        data = {}
        if title:
            data["title"] = title
        if state:
            data["state"] = state
        if priority:
            data["priority"] = priority
        if owner:
            data["owner"] = owner
        if group:
            data["group"] = group
            
        r = await client.put(f"{ZAMMAD_URL}/api/v1/tickets/{ticket_id}", 
                            headers=get_headers(), json=data)
        return r.json()

@mcp.tool()
async def zammad_add_article(
    ticket_id: int,
    body: str,
    article_type: ArticleType = ArticleType.NOTE,
    internal: bool = False,
    subject: str | None = None
) -> dict:
    """Add article to ticket (use internal ID, not display number)"""
    async with httpx.AsyncClient() as client:
        data = {
            "ticket_id": ticket_id,
            "body": body,
            "type": article_type.value,
            "internal": internal
        }
        if subject:
            data["subject"] = subject
            
        r = await client.post(f"{ZAMMAD_URL}/api/v1/ticket_articles", 
                             headers=get_headers(), json=data)
        return r.json()

@mcp.tool()
async def zammad_get_article_attachments(ticket_id: int, article_id: int) -> dict:
    """Get attachments for ticket article (use internal ID, not display number)"""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ZAMMAD_URL}/api/v1/ticket_articles/{article_id}/attachments",
            headers=get_headers()
        )
        return r.json()

@mcp.tool()
async def zammad_add_ticket_tag(ticket_id: int, tag: str) -> dict:
    """Add tag to ticket (use internal ID, not display number)"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/tags/add", 
                            headers=get_headers(), 
                            params={"object": "Ticket", "o_id": ticket_id, "item": tag})
        return r.json()

@mcp.tool()
async def zammad_remove_ticket_tag(ticket_id: int, tag: str) -> dict:
    """Remove tag from ticket (use internal ID, not display number)"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/tags/remove", 
                            headers=get_headers(), 
                            params={"object": "Ticket", "o_id": ticket_id, "item": tag})
        return r.json()

# Users
@mcp.tool()
async def zammad_get_user(user_id: int) -> dict:
    """Get user by ID"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/users/{user_id}", 
                            headers=get_headers())
        return r.json()

@mcp.tool()
async def zammad_search_users(
    query: str,
    page: int = 1,
    per_page: int = 10,
    response_format: ResponseFormat = ResponseFormat.MARKDOWN
) -> dict:
    """Search users"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/users/search", 
                            headers=get_headers(), 
                            params={"query": query, "page": page, "per_page": per_page})
        return r.json()

@mcp.tool()
async def zammad_get_current_user() -> dict:
    """Get currently authenticated user"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/users/me", 
                            headers=get_headers())
        return r.json()

# Organizations
@mcp.tool()
async def zammad_get_organization(org_id: int) -> dict:
    """Get organization by ID"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/organizations/{org_id}", 
                            headers=get_headers())
        return r.json()

@mcp.tool()
async def zammad_search_organizations(
    query: str,
    page: int = 1,
    per_page: int = 10,
    response_format: ResponseFormat = ResponseFormat.MARKDOWN
) -> dict:
    """Search organizations"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/organizations/search", 
                            headers=get_headers(), 
                            params={"query": query, "page": page, "per_page": per_page})
        return r.json()

# System
@mcp.tool()
async def zammad_list_groups(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> dict:
    """List all groups"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/groups", 
                            headers=get_headers())
        return r.json()

@mcp.tool()
async def zammad_list_ticket_states(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> dict:
    """List all ticket states"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/ticket_states", 
                            headers=get_headers())
        return r.json()

@mcp.tool()
async def zammad_list_ticket_priorities(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> dict:
    """List all ticket priorities"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/ticket_priorities", 
                            headers=get_headers())
        return r.json()

# Resources
@mcp.resource("zammad://ticket/{ticket_id}")
async def get_ticket_resource(ticket_id: str) -> str:
    """Get ticket as resource"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/tickets/{ticket_id}", 
                            headers=get_headers())
        ticket = r.json()
        
        lines = [
            f"Ticket #{ticket.get('number')} - {ticket.get('title')}",
            f"ID: {ticket.get('id')}",
            f"State: {ticket.get('state')}",
            f"Priority: {ticket.get('priority')}",
            f"Created: {ticket.get('created_at')}",
        ]
        return "\n".join(lines)

@mcp.resource("zammad://user/{user_id}")
async def get_user_resource(user_id: str) -> str:
    """Get user as resource"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/users/{user_id}", 
                            headers=get_headers())
        user = r.json()
        
        lines = [
            f"User: {user.get('firstname', '')} {user.get('lastname', '')}",
            f"Email: {user.get('email', '')}",
            f"Login: {user.get('login', '')}",
            f"Active: {user.get('active', False)}",
        ]
        return "\n".join(lines)

@mcp.resource("zammad://organization/{org_id}")
async def get_organization_resource(org_id: str) -> str:
    """Get organization as resource"""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ZAMMAD_URL}/api/v1/organizations/{org_id}", 
                            headers=get_headers())
        org = r.json()
        
        lines = [
            f"Organization: {org.get('name', '')}",
            f"Active: {org.get('active', False)}",
            f"Created: {org.get('created_at', 'Unknown')}",
        ]
        return "\n".join(lines)

if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http",
            host="0.0.0.0",
            port=12010,
        )
    )