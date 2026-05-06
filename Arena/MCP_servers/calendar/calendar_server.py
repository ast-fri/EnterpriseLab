# ============================================================================
# ENTERPRISE CALENDAR MCP SERVER - HYBRID APPROACH
# Storage + Background API Sync
# ============================================================================

import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from pydantic import Field
from calendar_storage import CalendarStorage
from aiohttp import ClientSession
from datetime import datetime, timedelta


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calendar_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# MCP SERVER INITIALIZATION
# ============================================================================

mcp = FastMCP(name="Enterprise Calendar MCP Server")


# ============================================================================
# STORAGE & CLIENT INITIALIZATION
# ============================================================================

storage = CalendarStorage()


class CalendarAPIClient:
    """HTTP Client for background API sync"""
    
    def __init__(self, base_url: str = "http://localhost:6000"):
        self.session: Optional[ClientSession] = None
        self.base_url = base_url.rstrip('/')

    async def init_session(self):
        """Initialize aiohttp session if not already initialized"""
        if self.session is None:
            self.session = ClientSession()

    async def api_request(
        self, 
        endpoint: str, 
        method: str = "GET", 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make API request to Calendar Backend
        
        Used for background sync - failures don't affect MCP operation
        """
        await self.init_session()
        url = f"{self.base_url}/api/{endpoint.lstrip('/')}"

        try:
            async with self.session.request(method, url, json=data) as response:
                response_data = await response.json()
                if response.status == 200:
                    return response_data
                else:
                    logger.warning(f"API {response.status}: {response_data}")
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            logger.warning(f"API sync failed: {str(e)}")
            return {"error": str(e)}


calendar_api_client = CalendarAPIClient(base_url="http://localhost:6000")


# ============================================================================
# TOOL 1: CREATE EVENT - HYBRID
# ============================================================================

@mcp.tool(
    name="create_event",
    description="Creates a new calendar event with title, time, description, and location."
)
async def create_event(
    title: str = Field(description="Event title"),
    start_time: str = Field(description="Event start time (ISO format: 2025-10-30T10:00:00)"),
    end_time: str = Field(description="Event end time (ISO format: 2025-10-30T11:00:00)"),
    organizer_email: str = Field(description="Organizer email address"),
    description: Optional[str] = Field(default="", description="Event description"),
    location: Optional[str] = Field(default="", description="Event location"),
    attendees: Optional[List[str]] = Field(default=None, description="List of attendee emails"),
    color: Optional[str] = Field(default="blue", description="Event color")
) -> Dict[str, Any]:
    """Create event"""
    
    # STEP 1: Local storage operations
    events = storage.load_json(storage.events_path)
    event_id = storage.generate_event_id()
    
    new_event = {
        "event_id": event_id,
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "description": description,
        "location": location,
        "organizer_email": organizer_email,
        "attendees": attendees or [],
        "color": color,
        "created_at": storage.get_current_datetime(),
        "updated_at": storage.get_current_datetime(),
        "status": "scheduled"
    }
    
    events.append(new_event)
    storage.save_json(storage.events_path, events)
    
    # STEP 2: Background API sync (fire-and-forget)
    asyncio.create_task(
        calendar_api_client.api_request(
            endpoint="events",
            method="POST",
            data={
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "location": location,
                "organizer_email": organizer_email,
                "attendees": attendees,
                "color": color
            }
        )
    )
    
    logger.info(f"✅ Event created: {event_id} - {title}")
    return {
        "status": "success",
        "event_id": event_id,
        "message": f"Event '{title}' created successfully"
    }


# ============================================================================
# TOOL 2: GET EVENT
# ============================================================================

@mcp.tool(
    name="get_event",
    description="Retrieves event details by event ID."
)
async def get_event(
    event_id: str = Field(description="Event ID to retrieve")
) -> Dict[str, Any]:
    """Get event details"""
    
    events = storage.load_json(storage.events_path)
    
    for event in events:
        if event.get("event_id") == event_id:
            logger.info(f"📖 Retrieved event: {event_id}")
            return {
                "status": "success",
                "event": event
            }
    
    logger.warning(f"❌ Event not found: {event_id}")
    return {
        "status": "error",
        "message": f"Event {event_id} not found"
    }


# ============================================================================
# TOOL 3: UPDATE EVENT - HYBRID
# ============================================================================

@mcp.tool(
    name="update_event",
    description="Updates an existing event (title, time, description, location)."
)
async def update_event(
    event_id: str = Field(description="Event ID to update"),
    title: Optional[str] = Field(default=None, description="New title"),
    start_time: Optional[str] = Field(default=None, description="New start time"),
    end_time: Optional[str] = Field(default=None, description="New end time"),
    description: Optional[str] = Field(default=None, description="New description"),
    location: Optional[str] = Field(default=None, description="New location"),
    color: Optional[str] = Field(default=None, description="New color"),
    status: Optional[str] = Field(default=None, description="New status")
) -> Dict[str, Any]:
    """Update event"""
    
    # STEP 1: Local storage operations
    events = storage.load_json(storage.events_path)
    
    for event in events:
        if event.get("event_id") == event_id:
            # Update only provided fields
            if title is not None:
                event["title"] = title
            if start_time is not None:
                event["start_time"] = start_time
            if end_time is not None:
                event["end_time"] = end_time
            if description is not None:
                event["description"] = description
            if location is not None:
                event["location"] = location
            if color is not None:
                event["color"] = color
            if status is not None:
                event["status"] = status
            
            event["updated_at"] = storage.get_current_datetime()
            storage.save_json(storage.events_path, events)
            
            # STEP 2: Background API sync
            update_data = {"event_id": event_id}
            if title is not None:
                update_data["title"] = title
            if start_time is not None:
                update_data["start_time"] = start_time
            if end_time is not None:
                update_data["end_time"] = end_time
            if description is not None:
                update_data["description"] = description
            if location is not None:
                update_data["location"] = location
            if color is not None:
                update_data["color"] = color
            if status is not None:
                update_data["status"] = status
            
            asyncio.create_task(
                calendar_api_client.api_request(
                    endpoint=f"events/{event_id}",
                    method="PUT",
                    data=update_data
                )
            )
            
            logger.info(f"✏️ Event updated: {event_id}")
            return {
                "status": "success",
                "event_id": event_id,
                "message": "Event updated successfully"
            }
    
    logger.warning(f"❌ Event not found for update: {event_id}")
    return {
        "status": "error",
        "message": f"Event {event_id} not found"
    }


# ============================================================================
# TOOL 4: DELETE EVENT - HYBRID
# ============================================================================

@mcp.tool(
    name="delete_event",
    description="Deletes a calendar event."
)
async def delete_event(
    event_id: str = Field(description="Event ID to delete")
) -> Dict[str, Any]:
    """Delete event"""
    
    # STEP 1: Local storage operations
    events = storage.load_json(storage.events_path)
    
    for i, event in enumerate(events):
        if event.get("event_id") == event_id:
            deleted_event = events.pop(i)
            storage.save_json(storage.events_path, events)
            
            # STEP 2: Background API sync
            asyncio.create_task(
                calendar_api_client.api_request(
                    endpoint=f"events/{event_id}",
                    method="DELETE"
                )
            )
            
            logger.info(f"🗑️ Event deleted: {event_id}")
            return {
                "status": "success",
                "message": f"Event '{deleted_event.get('title')}' deleted"
            }
    
    logger.warning(f"❌ Event not found for deletion: {event_id}")
    return {
        "status": "error",
        "message": f"Event {event_id} not found"
    }


# ============================================================================
# TOOL 5: LIST EVENTS - HYBRID
# ============================================================================

@mcp.tool(
    name="list_events",
    description="Lists events within a specified time range."
)
async def list_events(
    start_date: str = Field(description="Start date (ISO format: 2025-10-30)"),
    end_date: str = Field(description="End date (ISO format: 2025-10-31)"),
    organizer_email: Optional[str] = Field(default=None, description="Filter by organizer email"),
    attendee_email: Optional[str] = Field(default=None, description="Filter by attendee email")
) -> List[Dict[str, Any]]:
    """List events"""
    
    # STEP 1: Local storage search
    events = storage.load_json(storage.events_path)
    results = []
    
    # Parse dates
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
    except ValueError:
        logger.error(f"Invalid date format: {start_date} or {end_date}")
        return []
    
    for event in events:
        try:
            event_start = datetime.fromisoformat(event.get("start_time", "").replace('Z', '+00:00'))
            
            # Check if event is within date range
            if start <= event_start <= end:
                # Apply email filters if provided
                if organizer_email and event.get("organizer_email") != organizer_email:
                    continue
                if attendee_email and attendee_email not in event.get("attendees", []):
                    continue
                
                results.append({
                    "event_id": event.get("event_id"),
                    "title": event.get("title"),
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "location": event.get("location"),
                    "organizer_email": event.get("organizer_email"),
                    "attendees": event.get("attendees", []),
                    "color": event.get("color"),
                    "status": event.get("status"),
                    "has_attachments": False
                })
        except ValueError as e:
            logger.warning(f"Could not parse event date: {event.get('start_time')} - {e}")
            continue
    
    logger.info(f"📅 Found {len(results)} events between {start_date} and {end_date}")
    
    # STEP 2: Background API sync
    asyncio.create_task(
        calendar_api_client.api_request(
            endpoint="events/list",
            method="GET",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "organizer_email": organizer_email
            }
        )
    )
    
    return results


# ============================================================================
# TOOL 6: SEARCH EVENTS - HYBRID
# ============================================================================

@mcp.tool(
    name="search_events",
    description="Search events by title, location, or description."
)
async def search_events(
    query: str = Field(description="Search query"),
    max_results: int = Field(default=10, description="Maximum results")
) -> List[Dict[str, Any]]:
    """Search events"""
    
    # STEP 1: Local storage search
    events = storage.load_json(storage.events_path)
    results = []
    query_lower = query.lower()
    
    for event in events:
        # Search in title, location, and description
        if (query_lower in event.get("title", "").lower() or
            query_lower in event.get("location", "").lower() or
            query_lower in event.get("description", "").lower()):
            
            results.append({
                "event_id": event.get("event_id"),
                "title": event.get("title"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "location": event.get("location"),
                "organizer_email": event.get("organizer_email")
            })
        
        if len(results) >= max_results:
            break
    
    logger.info(f"🔍 Found {len(results)} events matching '{query}'")
    
    # STEP 2: Background API sync
    asyncio.create_task(
        calendar_api_client.api_request(
            endpoint="events/search",
            method="POST",
            data={
                "query": query,
                "max_results": max_results
            }
        )
    )
    
    return results


# ============================================================================
# TOOL 7: GET USER EVENTS - HYBRID
# ============================================================================

@mcp.tool(
    name="get_user_events",
    description="Get all events for a specific user (as organizer or attendee)."
)
async def get_user_events(
    user_email: str = Field(description="User email address"),
    status: Optional[str] = Field(default=None, description="Filter by status")
) -> List[Dict[str, Any]]:
    """Get user events"""
    
    # STEP 1: Local storage search
    events = storage.load_json(storage.events_path)
    results = []
    
    for event in events:
        # Check if user is organizer or attendee
        is_organizer = event.get("organizer_email") == user_email
        is_attendee = user_email in event.get("attendees", [])
        
        if is_organizer or is_attendee:
            if status and event.get("status") != status:
                continue
            
            results.append({
                "event_id": event.get("event_id"),
                "title": event.get("title"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "location": event.get("location"),
                "organizer_email": event.get("organizer_email"),
                "attendees": event.get("attendees", []),
                "status": event.get("status"),
                "role": "organizer" if is_organizer else "attendee"
            })
    
    logger.info(f"📅 Retrieved {len(results)} events for {user_email}")
    
    # STEP 2: Background API sync
    asyncio.create_task(
        calendar_api_client.api_request(
            endpoint=f"users/{user_email}/events",
            method="GET",
            params={"status": status} if status else None
        )
    )
    
    return results


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ENTERPRISE CALENDAR MCP SERVER")
    print("=" * 80)
    print("📡 Mode: Hybrid (Local Storage + Background API Sync)")
    print("🏠 Storage: Local JSON")
    print("☁️  Backend: http://localhost:6000")
    print("=" * 80)
    
    try:
        asyncio.run(
            mcp.run_async(
                transport="streamable-http",
                host="0.0.0.0",
                port=12006,
            )
        )
    except KeyboardInterrupt:
        print("\\n📴 Shutting down Calendar MCP Server...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise