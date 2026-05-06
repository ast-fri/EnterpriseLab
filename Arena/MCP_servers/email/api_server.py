from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from EnterpriseLab.Arena.MCP_servers.email.mcp_client_core import email_client
import uvicorn

# ============================================================================
# LIFESPAN EVENTS (Modern FastAPI approach)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        email_client.connect_to_server("server.py")
        print("✅ MCP Client connected successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to MCP server: {e}")
    
    yield
    
    # Shutdown
    try:
        email_client.disconnect()
        print("👋 MCP Client disconnected")
    except Exception as e:
        print(f"⚠️ Warning during disconnect: {e}")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Enterprise Email API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3002", 
        "http://localhost:5173",
        "http://localhost:7000"  # Add this,
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EmailSend(BaseModel):
    to: str
    cc: Optional[str] = ""
    subject: str
    body: str
    sender_email: Optional[str] = ""
    importance: str = "Normal"

class EmailSearch(BaseModel):
    query: str
    max_results: int = 50

class LabelModify(BaseModel):
    message_id: str
    add_labels: Optional[List[str]] = None
    remove_labels: Optional[List[str]] = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Enterprise Email API"}

@app.get("/api/status")
async def get_status():
    """Get connection status"""
    try:
        connected = email_client._connected
        num_tools = len(email_client.tools)
        return {
            "connected": connected,
            "num_tools": num_tools,
            "message": f"Connected with {num_tools} tools" if connected else "Not connected"
        }
    except Exception as e:
        return {"connected": False, "num_tools": 0, "message": str(e)}

@app.post("/api/connect")
async def connect():
    """Connect to MCP server"""
    try:
        num_tools = email_client.connect_to_server("server.py")
        return {
            "success": True,
            "connected": True,
            "num_tools": num_tools,
            "message": f"Connected with {num_tools} tools"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/send")
async def send_email(email: EmailSend):
    """Send an email"""
    try:
        to_list = [e.strip() for e in email.to.split(",") if e.strip()]
        cc_list = [e.strip() for e in email.cc.split(",") if e.strip() and email.cc]
        
        result = email_client.call_tool("send_email", {
            "to": to_list,
            "subject": email.subject,
            "body": email.body,
            "cc": cc_list if cc_list else None,
            "sender_email": email.sender_email if email.sender_email else None,
            "importance": email.importance
        })
        
        result_text = str(result.content[0].text) if hasattr(result, 'content') else str(result)
        
        return {
            "success": True,
            "message": "Email sent successfully",
            "data": result_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/draft")
async def draft_email(email: EmailSend):
    """Save email as draft"""
    try:
        to_list = [e.strip() for e in email.to.split(",") if e.strip()]
        cc_list = [e.strip() for e in email.cc.split(",") if e.strip() and email.cc]
        
        result = email_client.call_tool("draft_email", {
            "to": to_list,
            "subject": email.subject,
            "body": email.body,
            "cc": cc_list if cc_list else None,
            "sender_email": email.sender_email if email.sender_email else None
        })
        
        result_text = str(result.content[0].text) if hasattr(result, 'content') else str(result)
        
        return {
            "success": True,
            "message": "Draft saved successfully",
            "data": result_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/search")
async def search_emails(search: EmailSearch):
    """Search emails"""
    try:
        print(search)
        result = email_client.call_tool("search_emails", {
            "query": search.query,
            "maxResults": search.max_results
        })
        
        # The result might be a list of dicts, not text
        if hasattr(result, 'content') and len(result.content) > 0:
            content = result.content[0]
            
            # Check if it's text or already structured data
            if hasattr(content, 'text'):
                result_data = content.text
            else:
                result_data = str(content)
        else:
            result_data = []
        
        print(f"🔍 Search result type: {type(result_data)}")
        print(f"🔍 Search result: {result_data}")
        
        return {
            "success": True,
            "data": result_data,
            "message": "Search completed"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Search error: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emails/{message_id}")
async def read_email(message_id: str):
    """Read an email"""
    try:
        result = email_client.call_tool("read_email", {
            "messageId": message_id
        })
        
        result_text = str(result.content[0].text) if hasattr(result, 'content') else str(result)
        
        return {
            "success": True,
            "data": result_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/emails/{message_id}")
async def delete_email(message_id: str):
    """Delete an email"""
    try:
        result = email_client.call_tool("delete_email", {
            "messageId": message_id
        })
        
        return {
            "success": True,
            "message": "Email deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/labels")
async def list_labels():
    """List all labels"""
    try:
        result = email_client.call_tool("list_email_labels", {})
        
        result_text = str(result.content[0].text) if hasattr(result, 'content') else str(result)
        
        return {
            "success": True,
            "data": result_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/labels")
async def create_label(label: Dict[str, str]):
    """Create a new label"""
    try:
        result = email_client.call_tool("create_label", {
            "name": label.get("name")
        })
        
        return {
            "success": True,
            "message": "Label created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/emails/{message_id}/labels")
async def modify_labels(message_id: str, labels: LabelModify):
    """Modify email labels"""
    try:
        result = email_client.call_tool("modify_email", {
            "messageId": message_id,
            "addLabelIds": labels.add_labels,
            "removeLabelIds": labels.remove_labels
        })
        
        return {
            "success": True,
            "message": "Labels modified successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/emails/{message_id}/star")
async def toggle_star(message_id: str):
    """Toggle star on email"""
    try:
        result = email_client.call_tool("modify_email", {
            "messageId": message_id,
            "addLabelIds": ["STARRED"],
            "removeLabelIds": None
        })
        
        return {
            "success": True,
            "message": "Email starred"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add after existing endpoints

@app.post("/api/attachments/download")
async def download_attachment(data: Dict[str, Any]):
    """Download attachment"""
    try:
        result = email_client.call_tool("download_attachment", {
            "messageId": data.get("messageId"),
            "attachmentId": data.get("attachmentId"),
            "savePath": data.get("savePath", "."),
            "filename": data.get("filename")
        })
        
        return {
            "success": True,
            "data": str(result)
        }
    except Exception as e:
        import traceback
        print(f"❌ Download attachment error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/labels/{label_id}")
async def update_label(label_id: str, label: Dict[str, Any]):
    """Update a label"""
    try:
        result = email_client.call_tool("update_label", {
            "id": label_id,
            "name": label.get("name"),
            "messageListVisibility": label.get("messageListVisibility"),
            "labelListVisibility": label.get("labelListVisibility")
        })
        
        return {
            "success": True,
            "message": "Label updated successfully"
        }
    except Exception as e:
        import traceback
        print(f"❌ Update label error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/labels/{label_id}")
async def delete_label_endpoint(label_id: str):
    """Delete a label"""
    try:
        result = email_client.call_tool("delete_label", {
            "id": label_id
        })
        
        return {
            "success": True,
            "message": "Label deleted successfully"
        }
    except Exception as e:
        import traceback
        print(f"❌ Delete label error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/labels/get-or-create")
async def get_or_create_label_endpoint(label: Dict[str, str]):
    """Get or create a label"""
    try:
        result = email_client.call_tool("get_or_create_label", {
            "name": label.get("name")
        })
        
        return {
            "success": True,
            "data": str(result)
        }
    except Exception as e:
        import traceback
        print(f"❌ Get or create label error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/batch-modify")
async def batch_modify_emails_endpoint(data: Dict[str, Any]):
    """Batch modify emails"""
    try:
        result = email_client.call_tool("batch_modify_emails", {
            "messageIds": data.get("messageIds", []),
            "addLabelIds": data.get("add_labels"),
            "removeLabelIds": data.get("remove_labels")
        })
        
        return {
            "success": True,
            "data": str(result)
        }
    except Exception as e:
        import traceback
        print(f"❌ Batch modify error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/batch-delete")
async def batch_delete_emails_endpoint(data: Dict[str, Any]):
    """Batch delete emails"""
    try:
        result = email_client.call_tool("batch_delete_emails", {
            "messageIds": data.get("messageIds", [])
        })
        
        return {
            "success": True,
            "data": str(result)
        }
    except Exception as e:
        import traceback
        print(f"❌ Batch delete error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/filters")
async def list_filters_endpoint():
    """List all filters"""
    try:
        result = email_client.call_tool("list_filters", {})
        
        if hasattr(result, 'content') and len(result.content) > 0:
            result_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
        else:
            result_text = str(result)
        
        return {
            "success": True,
            "data": result_text
        }
    except Exception as e:
        import traceback
        print(f"❌ List filters error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filters")
async def create_filter_endpoint(data: Dict[str, Any]):
    """Create a filter"""
    try:
        result = email_client.call_tool("create_filter", {
            "criteria": data.get("criteria", {}),
            "action": data.get("action", {})
        })
        
        return {
            "success": True,
            "data": str(result)
        }
    except Exception as e:
        import traceback
        print(f"❌ Create filter error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filters/template")
async def create_filter_from_template_endpoint(data: Dict[str, Any]):
    """Create filter from template"""
    try:
        result = email_client.call_tool("create_filter_from_template", {
            "template": data.get("template"),
            "parameters": data.get("parameters", {})
        })
        
        return {
            "success": True,
            "data": str(result)
        }
    except Exception as e:
        import traceback
        print(f"❌ Create filter from template error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/filters/{filter_id}")
async def delete_filter_endpoint(filter_id: str):
    """Delete a filter"""
    try:
        result = email_client.call_tool("delete_filter", {
            "filterId": filter_id
        })
        
        return {
            "success": True,
            "message": "Filter deleted successfully"
        }
    except Exception as e:
        import traceback
        print(f"❌ Delete filter error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Enterprise Email API Server...")
    print("📧 API Documentation: http://localhost:7000/docs")
    print("🔗 React Frontend: http://localhost:3002")
    
    uvicorn.run(
        "api_server:app",  # Changed from app to "api_server:app" string
        host="0.0.0.0",
        port=7000,
        reload=True
    )
