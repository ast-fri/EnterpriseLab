#!/usr/bin/env python3
"""
OwnCloud MCP Server - HTTP Version
Provides tools to interact with a self-hosted OwnCloud instance via HTTP
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List
from urllib.parse import urljoin
import aiohttp
from fastmcp import FastMCP
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("owncloud-mcp")

class OwnCloudMCPServer:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.session = None
        self.dav_base = f"/remote.php/dav/files/{username}"
    
    def _normalize_path(self, path: str) -> str:
        """Remove WebDAV base prefix if present to handle both path formats"""
        if path.startswith(self.dav_base):
            return path[len(self.dav_base):]
        return path
        
    async def _get_session(self):
        if self.session is None:
            auth = aiohttp.BasicAuth(self.username, self.password)
            self.session = aiohttp.ClientSession(auth=auth)
        return self.session
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        session = await self._get_session()
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = await session.request(method, url, **kwargs)
            if response.status == 401:
                raise Exception("Authentication failed. Check credentials.")
            elif response.status >= 500:
                raise Exception(f"Server error {response.status}")
            return response
        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def _parse_propfind(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parse WebDAV PROPFIND response"""
        try:
            root = ET.fromstring(xml_data)
            ns = {'d': 'DAV:'}
            files = []
            
            for response in root.findall('.//d:response', ns):
                href = response.find('d:href', ns)
                propstat = response.find('d:propstat', ns)
                
                if href is not None and propstat is not None:
                    prop = propstat.find('d:prop', ns)
                    if prop is not None:
                        path = href.text
                        # Normalize the path to remove the DAV base
                        if path.startswith(self.dav_base):
                            path = path[len(self.dav_base):]
                        
                        name = path.rstrip('/').split('/')[-1] or '/'
                        
                        file_info = {
                            'path': path,  # Now this will be the clean path
                            'name': name,
                            'type': 'directory' if prop.find('d:resourcetype/d:collection', ns) is not None else 'file'
                        }
                        
                        size_elem = prop.find('d:getcontentlength', ns)
                        if size_elem is not None and size_elem.text:
                            file_info['size'] = int(size_elem.text)
                        
                        modified_elem = prop.find('d:getlastmodified', ns)
                        if modified_elem is not None:
                            file_info['modified'] = modified_elem.text
                        
                        files.append(file_info)
            
            return files[1:] if len(files) > 1 else files
        except Exception as e:
            logger.error(f"XML parse error: {e}")
            return []
    
    async def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """List files using WebDAV PROPFIND"""
        path = self._normalize_path(path)
        endpoint = f"{self.dav_base}{path}"
        
        propfind_body = '''<?xml version="1.0"?>
        <d:propfind xmlns:d="DAV:">
            <d:prop>
                <d:resourcetype/>
                <d:getcontentlength/>
                <d:getlastmodified/>
            </d:prop>
        </d:propfind>'''
        
        response = await self._make_request(
            "PROPFIND", 
            endpoint,
            headers={"Depth": "1", "Content-Type": "application/xml"},
            data=propfind_body
        )
        
        xml_data = await response.text()
        await response.release()
        return self._parse_propfind(xml_data)
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file info using WebDAV PROPFIND"""
        path = self._normalize_path(path)
        endpoint = f"{self.dav_base}{path}"
        
        propfind_body = '''<?xml version="1.0"?>
        <d:propfind xmlns:d="DAV:">
            <d:prop>
                <d:resourcetype/>
                <d:getcontentlength/>
                <d:getlastmodified/>
                <d:getcontenttype/>
            </d:prop>
        </d:propfind>'''
        
        response = await self._make_request(
            "PROPFIND",
            endpoint,
            headers={"Depth": "0", "Content-Type": "application/xml"},
            data=propfind_body
        )
        
        xml_data = await response.text()
        await response.release()
        files = self._parse_propfind(xml_data)
        return files[0] if files else {}
    
    async def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload file via WebDAV PUT"""
        remote_path = self._normalize_path(remote_path)
        endpoint = f"{self.dav_base}{remote_path}"
        
        try:
            with open(local_path, 'rb') as file:
                response = await self._make_request("PUT", endpoint, data=file)
                await response.release()
                return {"status": "success", "message": f"Uploaded to {remote_path}"}
        except FileNotFoundError:
            raise Exception(f"Local file not found: {local_path}")
    
    async def download_file(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """Download file via WebDAV GET"""
        remote_path = self._normalize_path(remote_path)
        endpoint = f"{self.dav_base}{remote_path}"
        
        response = await self._make_request("GET", endpoint)
        
        content_type = response.headers.get('Content-Type', '')
        
        if 'text/html' in content_type and response.status == 200:
            await response.release()
            raise Exception("Received HTML page instead of file. Check authentication or path.")
        
        try:
            with open(local_path, 'wb') as file:
                async for chunk in response.content.iter_chunked(8192):
                    file.write(chunk)
            await response.release()
            return {"status": "success", "message": f"Downloaded to {local_path}"}
        except Exception as e:
            await response.release()
            raise Exception(f"Download failed: {str(e)}")
        
    async def create_folder(self, path: str) -> Dict[str, Any]:
        """Create folder via WebDAV MKCOL"""
        path = self._normalize_path(path)
        endpoint = f"{self.dav_base}{path}"
        response = await self._make_request("MKCOL", endpoint)
        await response.release()
        return {"status": "success", "message": f"Created folder: {path}"}
    
    async def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete via WebDAV DELETE"""
        path = self._normalize_path(path)
        endpoint = f"{self.dav_base}{path}"
        response = await self._make_request("DELETE", endpoint)
        await response.release()
        return {"status": "success", "message": f"Deleted: {path}"}
    
    async def search_files(self, query: str, path: str = "/") -> List[Dict[str, Any]]:
        """Search files by name recursively"""
        results = []
        
        async def search_recursive(current_path: str):
            try:
                items = await self.list_files(current_path)
                for item in items:
                    if query.lower() in item.get("name", "").lower():
                        # Normalize the path to show from root
                        item_path = item.get("path", "")
                        if item_path.startswith(self.dav_base):
                            item["path"] = item_path[len(self.dav_base):]
                        results.append(item)
                    
                    if item.get("type") == "directory":
                        # Recursively search subdirectories
                        next_path = current_path.rstrip("/") + "/" + item.get("name")
                        await search_recursive(next_path)
            except Exception as e:
                logger.warning(f"Could not search in {current_path}: {e}")
        
        await search_recursive(path)
        return results
    
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage info via OCS API"""
        endpoint = "/ocs/v2.php/cloud/user"
        params = {"format": "json"}
        
        response = await self._make_request("GET", endpoint, params=params)
        text = await response.text()
        await response.release()
        
        try:
            data = json.loads(text)
            user_data = data.get("ocs", {}).get("data", {})
            
            # Extract quota if available
            quota = user_data.get("quota", {})
            if quota:
                user_data["quota_details"] = {
                    "free": quota.get("free"),
                    "used": quota.get("used"),
                    "total": quota.get("total"),
                    "relative": quota.get("relative")
                }
            
            return user_data
        except Exception as e:
            logger.error(f"Storage info error: {e}, Response: {text}")
            return {"error": "Could not retrieve storage info", "raw_response": text}

    async def read_file_content(self, remote_path: str) -> str:
        """Read and return the text content of a file"""
        remote_path = self._normalize_path(remote_path)
        endpoint = f"{self.dav_base}{remote_path}"
        
        response = await self._make_request("GET", endpoint)
        
        content_type = response.headers.get('Content-Type', '')
        
        if 'text/html' in content_type and response.status == 200:
            text = await response.text()
            await response.release()
            if 'untrusted domain' in text.lower() or 'body-login' in text:
                raise Exception("Cannot access file: untrusted domain error. Please configure trusted_domains in OwnCloud config.php")
            raise Exception("Received HTML page instead of file content.")
        
        try:
            content = await response.text()
            await response.release()
            return content
        except UnicodeDecodeError:
            await response.release()
            raise Exception("Cannot read file as text - it may be a binary file")
        except Exception as e:
            await response.release()
            raise Exception(f"Read failed: {str(e)}")

OWNCLOUD_URL = os.getenv("OWNCLOUD_URL", "http://localhost:8081")
OWNCLOUD_USERNAME = os.getenv("OWNCLOUD_USERNAME", "admin")
OWNCLOUD_PASSWORD = os.getenv("OWNCLOUD_PASSWORD", "admin")

owncloud = OwnCloudMCPServer(OWNCLOUD_URL, OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD)
mcp = FastMCP("OwnCloud MCP Server")

@mcp.tool()
async def list_files(path: str = "/") -> str:
    """List files and folders in OwnCloud"""
    try:
        result = await owncloud.list_files(path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_file_info(path: str) -> str:
    """Get detailed information about a file or folder"""
    try:
        result = await owncloud.get_file_info(path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_file_info: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def upload_file(local_path: str, remote_path: str) -> str:
    """Upload a file to OwnCloud"""
    try:
        result = await owncloud.upload_file(local_path, remote_path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def download_file(remote_path: str, local_path: str) -> str:
    """Download a file from OwnCloud"""
    try:
        result = await owncloud.download_file(remote_path, local_path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def create_folder(path: str) -> str:
    """Create a new folder in OwnCloud"""
    try:
        result = await owncloud.create_folder(path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in create_folder: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def delete_file(path: str) -> str:
    """Delete a file or folder from OwnCloud"""
    try:
        result = await owncloud.delete_file(path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in delete_file: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def search_files(query: str, path: str = "/") -> str:
    """Search for files by name recursively from the given path (default: root)"""
    try:
        result = await owncloud.search_files(query, path)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in search_files: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_storage_info() -> str:
    """Get storage quota and usage information"""
    try:
        result = await owncloud.get_storage_info()
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_storage_info: {str(e)}")
        return f"Error: {str(e)}"
        
@mcp.tool()
async def read_file_content(remote_path: str) -> str:
    """Read and return the text content of a file from OwnCloud"""
    try:
        result = await owncloud.read_file_content(remote_path)
        return result
    except Exception as e:
        logger.error(f"Error in read_file_content: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=12001,
        )
    )