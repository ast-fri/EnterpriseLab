from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import uvicorn
import asyncio
import threading
import time

# ============================================================================
# MCP CLIENT
# ============================================================================

class CalendarMCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = None
        self.tools = []
        self.loop = None
        self.thread = None
        self._connected = False
    
    def _start_background_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def ensure_loop(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._start_background_loop, daemon=True)
            self.thread.start()
            while self.loop is None:
                time.sleep(0.01)
    
    def connect_to_server(self, server_script_path: str = "calendar_server.py"):
        if self._connected:
            return len(self.tools)
        
        self.ensure_loop()
        
        async def do_connect():
            from contextlib import AsyncExitStack
            self.exit_stack = AsyncExitStack()
            
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
                env=None
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await self.session.initialize()
            response = await self.session.list_tools()
            
            # Properly extract tools
            if hasattr(response, 'tools'):
                self.tools = response.tools
            elif isinstance(response, dict) and 'tools' in response:
                self.tools = response['tools']
            else:
                self.tools = []
            
            self._connected = True
            
            print(f"📅 Loaded {len(self.tools)} calendar tools")
            for tool in self.tools:
                tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
                print(f"   - {tool_name}")
            
            return len(self.tools)
        
        future = asyncio.run_coroutine_threadsafe(do_connect(), self.loop)
        num_tools = future.result(timeout=30)
        return num_tools
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self._connected:
            raise Exception("Not connected to server")
        
        async def do_call():
            result = await self.session.call_tool(tool_name, arguments)
            return result
        
        future = asyncio.run_coroutine_threadsafe(do_call(), self.loop)
        return future.result(timeout=30)

calendar_client = CalendarMCPClient()

# ============================================================================
# FASTAPI SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        calendar_client.connect_to_server("calendar_server.py")
        print("✅ Calendar MCP Client connected")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Calendar MCP server: {e}")
    
    yield
    
    try:
        calendar_client._connected = False
        print("👋 Calendar MCP Client disconnected")
    except Exception as e:
        print(f"⚠️ Warning during disconnect: {e}")

app = FastAPI(
    title="Enterprise Calendar API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3002", 
        "http://localhost:5173",
        "http://localhost:7000"  # Add this,
        "http://localhost:6000"  # Add this,
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class CreateEvent(BaseModel):
    title: str
    start_time: str
    end_time: str
    description: Optional[str] = ""
    location: Optional[str] = ""
    organizer_email: str
    attendees: Optional[List[str]] = None
    color: Optional[str] = "blue"

class UpdateEvent(BaseModel):
    event_id: str
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    color: Optional[str] = None
    status: Optional[str] = None

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Enterprise Calendar API"}

@app.get("/api/status")
async def get_status():
    return {
        "connected": calendar_client._connected,
        "num_tools": len(calendar_client.tools),
        "message": f"Connected with {len(calendar_client.tools)} tools" if calendar_client._connected else "Not connected"
    }

@app.post("/api/connect")
async def connect():
    try:
        num_tools = calendar_client.connect_to_server("calendar_server.py")
        return {
            "success": True,
            "connected": True,
            "num_tools": num_tools,
            "message": f"Connected with {num_tools} tools"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ⭐ PUT THIS BEFORE get_event! ⭐
@app.get("/api/events/list")
async def list_events(start_date: str, end_date: str, organizer_email: Optional[str] = None):
    try:
        print(f"📅 Listing events from {start_date} to {end_date}")
        
        args = {
            "start_date": start_date, 
            "end_date": end_date
        }
        if organizer_email:
            args["organizer_email"] = organizer_email
        
        result = calendar_client.call_tool("list_events", args)
        result_data = ""
        if hasattr(result, 'content') and len(result.content) > 0:
            content = result.content[0]
            
            # Check if it's text or already structured data
            if hasattr(content, 'text'):
                result_data = content.text
            else:
                result_data = str(content)
        # print("Envent Result", result_data)
        
        
        return {"success": True, "data": result_data}
    except Exception as e:
        import traceback
        print(f"❌ List events error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ⭐ NOW put get_event after ⭐
@app.get("/api/events/{event_id}")
async def get_event(event_id: str):
    try:
        result = calendar_client.call_tool("get_event", {"event_id": event_id})
        result_data = ""
        if hasattr(result, 'content') and len(result.content) > 0:
            content = result.content[0]
            
            # Check if it's text or already structured data
            if hasattr(content, 'text'):
                result_data = content.text
            else:
                result_data = str(content)
        return {"success": True, "data": result_data}
    except Exception as e:
        import traceback
        print(f"❌ Get event error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/events")
async def create_event(event: CreateEvent):
    try:
        result = calendar_client.call_tool("create_event", {
            "title": event.title,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "description": event.description,
            "location": event.location,
            "organizer_email": event.organizer_email,
            "attendees": event.attendees,
            "color": event.color
        })
        return {"success": True, "data": result}
    except Exception as e:
        import traceback
        print(f"❌ Create event error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/events/{event_id}")
async def update_event(event_id: str, event: UpdateEvent):
    try:
        args = {"event_id": event_id}
        if event.title:
            args["title"] = event.title
        if event.start_time:
            args["start_time"] = event.start_time
        if event.end_time:
            args["end_time"] = event.end_time
        if event.description is not None:
            args["description"] = event.description
        if event.location is not None:
            args["location"] = event.location
        if event.color:
            args["color"] = event.color
        if event.status:
            args["status"] = event.status
        
        result = calendar_client.call_tool("update_event", args)
        return {"success": True, "data": result}
    except Exception as e:
        import traceback
        print(f"❌ Update event error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/events/{event_id}")
async def delete_event(event_id: str):
    try:
        result = calendar_client.call_tool("delete_event", {"event_id": event_id})
        return {"success": True, "data": result}
    except Exception as e:
        import traceback
        print(f"❌ Delete event error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_email}/events")
async def get_user_events(user_email: str, status: Optional[str] = None):
    try:
        args = {"user_email": user_email}
        if status:
            args["status"] = status
        
        result = calendar_client.call_tool("get_user_events", args)
        result_data = ""
        if hasattr(result, 'content') and len(result.content) > 0:
            content = result.content[0]
            
            # Check if it's text or already structured data
            if hasattr(content, 'text'):
                result_data = content.text
            else:
                result_data = str(content)
        return {"success": True, "data": result_data}
    except Exception as e:
        import traceback
        print(f"❌ Get user events error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/events/search")
async def search_events(query: str, max_results: int = 10):
    try:
        result = calendar_client.call_tool("search_events", {
            "query": query,
            "max_results": max_results
        })
        return {"success": True, "data": result}
    except Exception as e:
        import traceback
        print(f"❌ Search events error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Enterprise Calendar API Server...")
    print("📅 API Documentation: http://localhost:6000/docs")
    print("🔗 Calendar Port: 6000")
    
    uvicorn.run(
        "calendar_api_server:app",
        host="0.0.0.0",
        port=6000,
        reload=True
    )
