#!/usr/bin/env python3
"""
Rocket.Chat MCP Server - HTTP Version
A Model Context Protocol server for interacting with Rocket.Chat via HTTP
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
import aiohttp
from datetime import datetime
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RocketChatAPIClient:
    def __init__(self, rocketchat_url: str, user_id: str, auth_token: str):
        self.rocketchat_url = rocketchat_url.rstrip('/')
        self.user_id = user_id
        self.auth_token = auth_token
        self.session = None
        
        logger.info(f"Initialized Rocket.Chat client for {rocketchat_url}")
    
    async def _get_session(self):
        """Get or create aiohttp session with auth headers"""
        if self.session is None:
            headers = {
                'X-Auth-Token': self.auth_token,
                'X-User-Id': self.user_id,
                'Content-Type': 'application/json'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Rocket.Chat API"""
        session = await self._get_session()
        url = f"{self.rocketchat_url}/api/v1/{endpoint}"
        
        try:
            async with session.request(method, url, json=data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                return await response.json()
        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {str(e)}")
    
    # async def send_message(self, channel: str, message: str) -> Dict[str, Any]:
    #     """Send a message to a channel or direct message"""
    #     data = {
    #         "channel": channel,
    #         "text": message
    #     }
        
    #     result = await self.make_request("chat.postMessage", method="POST", data=data)
    #     return {
    #         "success": True,
    #         "message": f"Message sent successfully to {channel}: \"{message}\"",
    #         "data": result
    #     }


    async def send_channel_message(self, channel: str, message: str) -> Dict[str, Any]:
        """Send a message to a channel"""
        if not channel.startswith('#'):
            channel = f"#{channel}"
        
        data = {"channel": channel, "text": message}
        result = await self.make_request("chat.postMessage", method="POST", data=data)
        
        return {
            "success": True,
            "message": f"Message sent to channel {channel}: \"{message}\"",
            "data": result
        }

    async def send_direct_message(self, username: str, message: str) -> Dict[str, Any]:
        """Send a direct message to a user"""
        username = username.lstrip('@')
        
        try:
            dm_data = {"username": username}
            dm_result = await self.make_request("im.create", method="POST", data=dm_data)
            room_id = dm_result.get("room", {}).get("_id")
            
            data = {"channel": room_id, "text": message}
            result = await self.make_request("chat.postMessage", method="POST", data=data)
            
            return {
                "success": True,
                "message": f"Message sent to @{username}: \"{message}\"",
                "data": result
            }
        except Exception as e:
            raise Exception(f"Failed to send DM to {username}: {str(e)}")

    async def get_channels(self) -> Dict[str, Any]:
        """Get list of all available channels"""
        result = await self.make_request("channels.list")
        
        channels = []
        for ch in result.get("channels", []):
            channels.append({
                "name": ch.get("name"),
                "id": ch.get("_id"),
                "description": ch.get("description", ""),
                "userCount": ch.get("usersCount", 0)
            })
        
        return {
            "success": True,
            "channels": channels,
            "formatted": "\n".join([
                f"- #{ch['name']} ({ch['userCount']} users): {ch['description'] or 'No description'}"
                for ch in channels
            ])
        }
    
    async def get_channel_messages(self, channel: str, count: int = 20) -> Dict[str, Any]:
        """Get recent messages from a channel or direct message"""
        # Determine if it's a channel ID (starts with room ID pattern) or name
        if channel.startswith('#'):
            channel = channel[1:]  # Remove # prefix
            endpoint = f"channels.messages?roomName={channel}&count={count}"
        elif len(channel) == 17 or len(channel) == 24:  # Room ID length
            # Try to get room info first to determine type
            try:
                room_info = await self.make_request(f"rooms.info?roomId={channel}")
                room_type = room_info.get("room", {}).get("t", "c")
                
                if room_type == "d":  # Direct message
                    endpoint = f"im.messages?roomId={channel}&count={count}"
                elif room_type == "p":  # Private group
                    endpoint = f"groups.messages?roomId={channel}&count={count}"
                else:  # Public channel
                    endpoint = f"channels.messages?roomId={channel}&count={count}"
            except:
                # Fallback: try as channel first
                endpoint = f"channels.messages?roomId={channel}&count={count}"
        else:
            # Assume it's a channel name
            endpoint = f"channels.messages?roomName={channel}&count={count}"
        
        try:
            result = await self.make_request(endpoint)
        except Exception as e:
            # If channel endpoint fails, try as direct message
            if "roomName" in endpoint:
                return {"success": False, "error": str(e)}
            try:
                endpoint = f"im.messages?roomId={channel}&count={count}"
                result = await self.make_request(endpoint)
            except:
                try:
                    endpoint = f"groups.messages?roomId={channel}&count={count}"
                    result = await self.make_request(endpoint)
                except Exception as final_error:
                    return {"success": False, "error": str(final_error)}
        
        messages = []
        for msg in result.get("messages", []):
            user_info = msg.get("u", {})
            messages.append({
                "user": user_info.get("username", "Unknown"),
                "message": msg.get("msg", ""),
                "timestamp": msg.get("ts", "")
            })
        
        formatted_messages = []
        for msg in messages:
            if msg["timestamp"]:
                try:
                    if isinstance(msg["timestamp"], dict) and "$date" in msg["timestamp"]:
                        timestamp = datetime.fromtimestamp(msg["timestamp"]["$date"] / 1000)
                    else:
                        timestamp = datetime.fromisoformat(str(msg["timestamp"]).replace("Z", "+00:00"))
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = str(msg["timestamp"])
            else:
                formatted_time = "Unknown time"
            
            formatted_messages.append(f"[{formatted_time}] {msg['user']}: {msg['message']}")
        
        return {
            "success": True,
            "messages": messages,
            "formatted": f"Recent messages:\n" + "\n".join(formatted_messages)
        }
    
    async def create_channel(self, name: str, description: str = "", is_private: bool = False) -> Dict[str, Any]:
        """Create a new channel"""
        endpoint = "groups.create" if is_private else "channels.create"
        data = {
            "name": name,
            "description": description
        }
        
        result = await self.make_request(endpoint, method="POST", data=data)
        
        return {
            "success": True,
            "message": f"Channel \"{name}\" created successfully{' (private)' if is_private else ' (public)'}",
            "data": result
        }
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

# Initialize Rocket.Chat client from environment variables
ROCKETCHAT_URL = os.getenv("ROCKETCHAT_URL", "http://localhost:3000")
ROCKETCHAT_USER_ID = os.getenv("ROCKETCHAT_USER_ID","6BBQsLK5LRchYAM8f")
ROCKETCHAT_AUTH_TOKEN = os.getenv("ROCKETCHAT_AUTH_TOKEN","aW0jNxkyBfig7PaJ_FTcNE5nElwbbfEVRt6-WTKQSut")

if not ROCKETCHAT_USER_ID or not ROCKETCHAT_AUTH_TOKEN:
    raise ValueError("ROCKETCHAT_USER_ID and ROCKETCHAT_AUTH_TOKEN must be set")

rocketchat_client = RocketChatAPIClient(ROCKETCHAT_URL, ROCKETCHAT_USER_ID, ROCKETCHAT_AUTH_TOKEN)

# Create the FastMCP server
mcp = FastMCP("Rocket.Chat MCP Server")

# @mcp.tool()
# async def send_message(channel: str, message: str) -> str:
#     """Send a message to a Rocket.Chat channel or direct message"""
#     try:
#         result = await rocketchat_client.send_message(channel, message)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in send_message: {str(e)}")
#         return f"Error: {str(e)}"

@mcp.tool()
async def send_channel_message(channel: str, message: str) -> str:
    """Send a message to a channel (e.g., 'general', '#general')"""
    try:
        result = await rocketchat_client.send_channel_message(channel, message)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in send_channel_message: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def send_direct_message(username: str, message: str) -> str:
    """Send a direct message to a user (e.g., 'john.doe', '@john.doe')"""
    try:
        result = await rocketchat_client.send_direct_message(username, message)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in send_direct_message: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_channels() -> str:
    """List all available channels"""
    try:
        result = await rocketchat_client.get_channels()
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_channels: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_channel_messages(channel: str, count: int = 20) -> str:
    """Get recent messages from a channel"""
    try:
        result = await rocketchat_client.get_channel_messages(channel, count)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_channel_messages: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def create_channel(name: str, description: str = "", private: bool = False) -> str:
    """Create a new channel"""
    try:
        result = await rocketchat_client.create_channel(name, description, private)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in create_channel: {str(e)}")
        return f"Error: {str(e)}"

# Additional useful tools
@mcp.tool()
async def get_user_info() -> str:
    """Get information about the current user"""
    try:
        result = await rocketchat_client.make_request("me")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_user_info: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_server_info() -> str:
    """Get Rocket.Chat server information"""
    try:
        result = await rocketchat_client.make_request("info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_server_info: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def search_messages(query: str, room_id: str = None) -> str:
    """Search for messages"""
    try:
        params = f"searchText={query}"
        if room_id:
            params += f"&roomId={room_id}"
        
        result = await rocketchat_client.make_request(f"chat.search?{params}")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in search_messages: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_direct_messages() -> str:
    """Get list of direct message conversations"""
    try:
        result = await rocketchat_client.make_request("im.list")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_direct_messages: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_users() -> str:
    """Get list of users"""
    try:
        result = await rocketchat_client.make_request("users.list")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def join_channel(channel: str) -> str:
    """Join a channel"""
    try:
        data = {"roomName": channel}
        result = await rocketchat_client.make_request("channels.join", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in join_channel: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def leave_channel(channel: str) -> str:
    """Leave a channel"""
    try:
        data = {"roomName": channel}
        result = await rocketchat_client.make_request("channels.leave", method="POST", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in leave_channel: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=12004,
        )
    )

#!/usr/bin/env python3
# """
# Rocket.Chat MCP Server - HTTP Version
# A Model Context Protocol server for interacting with Rocket.Chat via HTTP
# """

# import os
# import json
# import asyncio
# import logging
# from typing import Any, Dict, List, Optional
# import aiohttp
# from datetime import datetime
# from fastmcp import FastMCP

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class RocketChatAPIClient:
#     def __init__(self, rocketchat_url: str, user_id: str, auth_token: str):
#         self.rocketchat_url = rocketchat_url.rstrip('/')
#         self.user_id = user_id
#         self.auth_token = auth_token
#         self.session = None
        
#         logger.info(f"Initialized Rocket.Chat client for {rocketchat_url}")
    
#     async def _get_session(self):
#         """Get or create aiohttp session with auth headers"""
#         if self.session is None:
#             headers = {
#                 'X-Auth-Token': self.auth_token,
#                 'X-User-Id': self.user_id,
#                 'Content-Type': 'application/json'
#             }
#             self.session = aiohttp.ClientSession(headers=headers)
#         return self.session
    
#     async def make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
#         """Make authenticated request to Rocket.Chat API"""
#         session = await self._get_session()
#         url = f"{self.rocketchat_url}/api/v1/{endpoint}"
        
#         try:
#             async with session.request(method, url, json=data) as response:
#                 if response.status >= 400:
#                     error_text = await response.text()
#                     raise Exception(f"API request failed with status {response.status}: {error_text}")
                
#                 return await response.json()
#         except aiohttp.ClientError as e:
#             raise Exception(f"Request failed: {str(e)}")
    
#     async def send_message(self, channel: str, message: str) -> Dict[str, Any]:
#         """Send a message to a channel or direct message
        
#         For direct messages to users, you can use:
#         - @username (e.g., @john.doe)
#         - Room ID (if you have it)
#         - Username without @ (will try to create/find DM)
#         """
#         # If it's a username (not starting with # or a room ID), ensure DM exists
#         if not channel.startswith('#') and not channel.startswith('@') and len(channel) < 17:
#             # It's likely a username, try to create DM first
#             try:
#                 dm_result = await self.create_direct_message(channel)
#                 if dm_result.get("success") and dm_result.get("room_id"):
#                     channel = dm_result["room_id"]
#             except Exception as e:
#                 logger.warning(f"Could not create/find DM with {channel}, trying with @ prefix: {e}")
#                 # Fallback: try with @ prefix
#                 if not channel.startswith('@'):
#                     channel = f"@{channel}"
        
#         data = {
#             "channel": channel,
#             "text": message
#         }
        
#         result = await self.make_request("chat.postMessage", method="POST", data=data)
#         return {
#             "success": True,
#             "message": f"Message sent successfully to {channel}: \"{message}\"",
#             "data": result
#         }
    
#     async def create_direct_message(self, username: str) -> Dict[str, Any]:
#         """Create or get existing direct message with a user
        
#         Args:
#             username: Username of the user (without @ prefix)
        
#         Returns:
#             Dict with success, room_id, and message
#         """
#         # Remove @ if present
#         username = username.lstrip('@')
        
#         data = {
#             "username": username
#         }
        
#         try:
#             result = await self.make_request("im.create", method="POST", data=data)
            
#             room_id = result.get("room", {}).get("_id")
#             if room_id:
#                 return {
#                     "success": True,
#                     "room_id": room_id,
#                     "message": f"Direct message channel with {username} is ready",
#                     "data": result
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "error": "Could not get room ID from response",
#                     "data": result
#                 }
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": f"Failed to create DM with {username}: {str(e)}"
#             }
    
#     async def get_channels(self) -> Dict[str, Any]:
#         """Get list of all available channels"""
#         result = await self.make_request("channels.list")
        
#         channels = []
#         for ch in result.get("channels", []):
#             channels.append({
#                 "name": ch.get("name"),
#                 "id": ch.get("_id"),
#                 "description": ch.get("description", ""),
#                 "userCount": ch.get("usersCount", 0)
#             })
        
#         return {
#             "success": True,
#             "channels": channels,
#             "formatted": "\n".join([
#                 f"- #{ch['name']} ({ch['userCount']} users): {ch['description'] or 'No description'}"
#                 for ch in channels
#             ])
#         }
    
#     async def get_channel_messages(self, channel: str, count: int = 20) -> Dict[str, Any]:
#         """Get recent messages from a channel or direct message"""
#         # Determine if it's a channel ID (starts with room ID pattern) or name
#         if channel.startswith('#'):
#             channel = channel[1:]  # Remove # prefix
#             endpoint = f"channels.messages?roomName={channel}&count={count}"
#         elif len(channel) == 17 or len(channel) == 24:  # Room ID length
#             # Try to get room info first to determine type
#             try:
#                 room_info = await self.make_request(f"rooms.info?roomId={channel}")
#                 room_type = room_info.get("room", {}).get("t", "c")
                
#                 if room_type == "d":  # Direct message
#                     endpoint = f"im.messages?roomId={channel}&count={count}"
#                 elif room_type == "p":  # Private group
#                     endpoint = f"groups.messages?roomId={channel}&count={count}"
#                 else:  # Public channel
#                     endpoint = f"channels.messages?roomId={channel}&count={count}"
#             except:
#                 # Fallback: try as channel first
#                 endpoint = f"channels.messages?roomId={channel}&count={count}"
#         else:
#             # Assume it's a channel name
#             endpoint = f"channels.messages?roomName={channel}&count={count}"
        
#         try:
#             result = await self.make_request(endpoint)
#         except Exception as e:
#             # If channel endpoint fails, try as direct message
#             if "roomName" in endpoint:
#                 return {"success": False, "error": str(e)}
#             try:
#                 endpoint = f"im.messages?roomId={channel}&count={count}"
#                 result = await self.make_request(endpoint)
#             except:
#                 try:
#                     endpoint = f"groups.messages?roomId={channel}&count={count}"
#                     result = await self.make_request(endpoint)
#                 except Exception as final_error:
#                     return {"success": False, "error": str(final_error)}
        
#         messages = []
#         for msg in result.get("messages", []):
#             user_info = msg.get("u", {})
#             messages.append({
#                 "user": user_info.get("username", "Unknown"),
#                 "message": msg.get("msg", ""),
#                 "timestamp": msg.get("ts", "")
#             })
        
#         formatted_messages = []
#         for msg in messages:
#             if msg["timestamp"]:
#                 try:
#                     if isinstance(msg["timestamp"], dict) and "$date" in msg["timestamp"]:
#                         timestamp = datetime.fromtimestamp(msg["timestamp"]["$date"] / 1000)
#                     else:
#                         timestamp = datetime.fromisoformat(str(msg["timestamp"]).replace("Z", "+00:00"))
#                     formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#                 except:
#                     formatted_time = str(msg["timestamp"])
#             else:
#                 formatted_time = "Unknown time"
            
#             formatted_messages.append(f"[{formatted_time}] {msg['user']}: {msg['message']}")
        
#         return {
#             "success": True,
#             "messages": messages,
#             "formatted": f"Recent messages:\n" + "\n".join(formatted_messages)
#         }
    
#     async def create_channel(self, name: str, description: str = "", is_private: bool = False) -> Dict[str, Any]:
#         """Create a new channel"""
#         endpoint = "groups.create" if is_private else "channels.create"
#         data = {
#             "name": name,
#             "description": description
#         }
        
#         result = await self.make_request(endpoint, method="POST", data=data)
        
#         return {
#             "success": True,
#             "message": f"Channel \"{name}\" created successfully{' (private)' if is_private else ' (public)'}",
#             "data": result
#         }
    
#     async def close(self):
#         """Close the session"""
#         if self.session:
#             await self.session.close()

# # Initialize Rocket.Chat client from environment variables
# ROCKETCHAT_URL = os.getenv("ROCKETCHAT_URL", "http://localhost:3000")
# ROCKETCHAT_USER_ID = os.getenv("ROCKETCHAT_USER_ID","6BBQsLK5LRchYAM8f")
# ROCKETCHAT_AUTH_TOKEN = os.getenv("ROCKETCHAT_AUTH_TOKEN","aW0jNxkyBfig7PaJ_FTcNE5nElwbbfEVRt6-WTKQSut")

# if not ROCKETCHAT_USER_ID or not ROCKETCHAT_AUTH_TOKEN:
#     raise ValueError("ROCKETCHAT_USER_ID and ROCKETCHAT_AUTH_TOKEN must be set")

# rocketchat_client = RocketChatAPIClient(ROCKETCHAT_URL, ROCKETCHAT_USER_ID, ROCKETCHAT_AUTH_TOKEN)

# # Create the FastMCP server
# mcp = FastMCP("Rocket.Chat MCP Server")

# @mcp.tool()
# async def send_message(channel: str, message: str) -> str:
#     """Send a message to a Rocket.Chat channel or direct message
    
#     Args:
#         channel: Can be channel name (#general), username (@user or user), or room ID
#         message: The message text to send
#     """
#     try:
#         result = await rocketchat_client.send_message(channel, message)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in send_message: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def create_direct_message(username: str) -> str:
#     """Create or get existing direct message conversation with a user
    
#     Args:
#         username: Username of the user (with or without @ prefix)
#     """
#     try:
#         result = await rocketchat_client.create_direct_message(username)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in create_direct_message: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def send_direct_message(username: str, message: str) -> str:
#     """Send a direct message to a user (creates DM if doesn't exist)
    
#     Args:
#         username: Username of the user (with or without @ prefix)
#         message: The message text to send
#     """
#     try:
#         # First ensure DM exists
#         dm_result = await rocketchat_client.create_direct_message(username)
        
#         if not dm_result.get("success"):
#             return json.dumps(dm_result, indent=2)
        
#         # Then send the message
#         room_id = dm_result.get("room_id")
#         result = await rocketchat_client.send_message(room_id, message)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in send_direct_message: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def get_channels() -> str:
#     """List all available channels"""
#     try:
#         result = await rocketchat_client.get_channels()
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in get_channels: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def get_channel_messages(channel: str, count: int = 20) -> str:
#     """Get recent messages from a channel"""
#     try:
#         result = await rocketchat_client.get_channel_messages(channel, count)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in get_channel_messages: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def create_channel(name: str, description: str = "", private: bool = False) -> str:
#     """Create a new channel"""
#     try:
#         result = await rocketchat_client.create_channel(name, description, private)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in create_channel: {str(e)}")
#         return f"Error: {str(e)}"

# # Additional useful tools
# @mcp.tool()
# async def get_user_info() -> str:
#     """Get information about the current user"""
#     try:
#         result = await rocketchat_client.make_request("me")
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in get_user_info: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def get_server_info() -> str:
#     """Get Rocket.Chat server information"""
#     try:
#         result = await rocketchat_client.make_request("info")
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in get_server_info: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def search_messages(query: str, room_id: str = None) -> str:
#     """Search for messages"""
#     try:
#         params = f"searchText={query}"
#         if room_id:
#             params += f"&roomId={room_id}"
        
#         result = await rocketchat_client.make_request(f"chat.search?{params}")
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in search_messages: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def get_direct_messages() -> str:
#     """Get list of direct message conversations"""
#     try:
#         result = await rocketchat_client.make_request("im.list")
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in get_direct_messages: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def get_users() -> str:
#     """Get list of users"""
#     try:
#         result = await rocketchat_client.make_request("users.list")
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in get_users: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def join_channel(channel: str) -> str:
#     """Join a channel"""
#     try:
#         data = {"roomName": channel}
#         result = await rocketchat_client.make_request("channels.join", method="POST", data=data)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in join_channel: {str(e)}")
#         return f"Error: {str(e)}"

# @mcp.tool()
# async def leave_channel(channel: str) -> str:
#     """Leave a channel"""
#     try:
#         data = {"roomName": channel}
#         result = await rocketchat_client.make_request("channels.leave", method="POST", data=data)
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error in leave_channel: {str(e)}")
#         return f"Error: {str(e)}"

# if __name__ == "__main__":
#     asyncio.run(
#         mcp.run_async(
#             transport="streamable-http", 
#             host="0.0.0.0", 
#             port=12004,
#         )
#     )