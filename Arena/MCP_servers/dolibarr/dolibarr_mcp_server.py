#!/usr/bin/env python3
"""
Dolibarr MCP Server - HTTP Version
A Model Context Protocol server for interacting with Dolibarr ERP/CRM via HTTP
"""

import json
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
import aiohttp
from datetime import datetime
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DolibarrAPIClient:
    def __init__(self, base_url: str = "http://localhost:8082", api_key: str = "", username: str = "", password: str = ""):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.session = None
        
        logger.info(f"Using API Key: {self.api_key}")
    
    async def init_session(self):
        """Initialize HTTP session with authentication"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
            # If API key is provided, use it
            if self.api_key:
                self.session.headers.update({
                    'DOLAPIKEY': self.api_key,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                })
            # Otherwise use username/password authentication
            elif self.username and self.password:
                login_data = {
                    'login': self.username,
                    'password': self.password
                }
                try:
                    async with self.session.post(f"{self.base_url}/api/index.php/login", json=login_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            if 'success' in result and result['success']:
                                # Set session token if provided
                                if 'token' in result:
                                    self.session.headers.update({
                                        'Authorization': f'Bearer {result["token"]}',
                                        'Accept': 'application/json',
                                        'Content-Type': 'application/json'
                                    })
                except Exception as e:
                    logger.error(f"Authentication failed: {e}")
    
    async def api_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated API request to Dolibarr"""
        await self.init_session()
        
        url = f"{self.base_url}/api/index.php/{endpoint}"
        
        try:
            async with self.session.request(method, url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

# Initialize Dolibarr client
DOLIBARR_BASE_URL = os.getenv("DOLIBARR_BASE_URL", "http://localhost:8082")
DOLIBARR_API_KEY = os.getenv("DOLIBARR_API_KEY", "9f68ccea83e23f5951cc6057df7c9e45f7363051")
DOLIBARR_USERNAME = os.getenv("DOLIBARR_USERNAME", "")
DOLIBARR_PASSWORD = os.getenv("DOLIBARR_PASSWORD", "")

dolibarr_client = DolibarrAPIClient(
    base_url=DOLIBARR_BASE_URL,
    api_key=DOLIBARR_API_KEY,
    username=DOLIBARR_USERNAME,
    password=DOLIBARR_PASSWORD
)

# Create the FastMCP server
mcp = FastMCP("Dolibarr MCP Server")

# Company tools
@mcp.tool()
async def get_companies(limit: int = None, sortfield: str = None, sortorder: str = None) -> str:
    """Get list of companies/third parties"""
    try:
        params = []
        if limit:
            params.append(f"limit={limit}")
        if sortfield:
            params.append(f"sortfield={sortfield}")
        if sortorder:
            params.append(f"sortorder={sortorder}")
        
        endpoint = "thirdparties"
        if params:
            endpoint += "?" + "&".join(params)
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_company(id: int) -> str:
    """Get specific company by ID"""
    try:
        result = await dolibarr_client.api_request(f"thirdparties/{id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_company(name: str, email: str = None, phone: str = None, address: str = None, zip: str = None, town: str = None, country_code: str = None) -> str:
    """Create a new company/third party"""
    try:
        data = {"name": name}
        if email:
            data["email"] = email
        if phone:
            data["phone"] = phone
        if address:
            data["address"] = address
        if zip:
            data["zip"] = zip
        if town:
            data["town"] = town
        if country_code:
            data["country_code"] = country_code
        
        result = await dolibarr_client.api_request("thirdparties", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Invoice tools
@mcp.tool()
async def get_invoices(thirdparty_id: int = None, status: str = None, limit: int = None) -> str:
    """Get list of invoices"""
    try:
        params = []
        if thirdparty_id:
            params.append(f"thirdparty_id={thirdparty_id}")
        if status:
            params.append(f"status={status}")
        if limit:
            params.append(f"limit={limit}")
        
        endpoint = "invoices"
        if params:
            endpoint += "?" + "&".join(params)
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_invoice(id: int) -> str:
    """Get specific invoice by ID"""
    try:
        result = await dolibarr_client.api_request(f"invoices/{id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_invoice(thirdparty_id: int, date: str = None, ref: str = None, note: str = None) -> str:
    """Create a new invoice"""
    try:
        data = {"thirdparty_id": thirdparty_id}
        if date:
            data["date"] = date
        if ref:
            data["ref"] = ref
        if note:
            data["note"] = note
        
        result = await dolibarr_client.api_request("invoices", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Product tools
@mcp.tool()
async def get_products(type: str = None, limit: int = None) -> str:
    """Get list of products/services"""
    try:
        params = []
        if type:
            params.append(f"type={type}")
        if limit:
            params.append(f"limit={limit}")
        
        endpoint = "products"
        if params:
            endpoint += "?" + "&".join(params)
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_product(id: int) -> str:
    """Get specific product by ID"""
    try:
        result = await dolibarr_client.api_request(f"products/{id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_product(ref: str, label: str, type: int = None, price: float = None, description: str = None) -> str:
    """Create a new product/service"""
    try:
        data = {"ref": ref, "label": label}
        if type is not None:
            data["type"] = type
        if price is not None:
            data["price"] = price
        if description:
            data["description"] = description
        
        result = await dolibarr_client.api_request("products", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Project tools
@mcp.tool()
async def get_projects(limit: int = None) -> str:
    """Get list of projects"""
    try:
        params = []
        if limit:
            params.append(f"limit={limit}")
        
        endpoint = "projects"
        if params:
            endpoint += "?" + "&".join(params)
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# User tools
@mcp.tool()
async def get_users(limit: int = None) -> str:
    """Get list of users (employees)"""
    try:
        params = []
        if limit:
            params.append(f"limit={limit}")
        
        endpoint = "users"
        if params:
            endpoint += "?" + "&".join(params)
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_user(id: int) -> str:
    """Get a specific user by ID"""
    try:
        result = await dolibarr_client.api_request(f"users/{id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_user(login: str, firstname: str, lastname: str, email: str, password: str, admin: int = 0) -> str:
    """Create a new user"""
    try:
        data = {
            "login": login,
            "firstname": firstname,
            "lastname": lastname,
            "email": email,
            "pass": password,
            "admin": admin
        }
        
        result = await dolibarr_client.api_request("users", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Holiday tools
@mcp.tool()
async def get_holidays(status: str = None) -> str:
    """Get list of leave/holiday requests"""
    try:
        endpoint = "holiday"
        if status:
            endpoint += f"?status={status}"
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def create_holiday_request(fk_user: int, date_debut: str, date_fin: str, type: str, halfday: int = 0, description: str = None) -> str:
    """Create a holiday/leave request"""
    try:
        data = {
            "fk_user": fk_user,
            "date_debut": date_debut,
            "date_fin": date_fin,
            "type": type,
            "halfday": halfday
        }
        if description:
            data["description"] = description
        
        result = await dolibarr_client.api_request("holiday", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Timesheet tools
@mcp.tool()
async def get_timesheets(fk_user: int = None, year: int = None, month: int = None) -> str:
    """Get user timesheets (activities)"""
    try:
        params = []
        if fk_user:
            params.append(f"fk_user={fk_user}")
        if month:
            params.append(f"month={month}")
        if year:
            params.append(f"year={year}")
        
        endpoint = "timespent/list"
        if params:
            endpoint += "?" + "&".join(params)
        
        result = await dolibarr_client.api_request(endpoint)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=12000,
        )
    )