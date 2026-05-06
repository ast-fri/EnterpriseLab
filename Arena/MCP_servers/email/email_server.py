# import asyncio
# from typing import List, Dict, Any, Optional, Literal
# from fastmcp import FastMCP
# from pydantic import Field
# from email_storage import EmailStorage
# import base64
# import os
# import logging
# # Configure logging to file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('server.log'),  # Write to file
#         logging.StreamHandler()  # Also print to console
#     ]
# )

# logger = logging.getLogger(__name__)
# # Initialize MCP server
# mcp = FastMCP(
#     name="Enterprise Email MCP Server"
# )

# # Initialize storage
# storage = EmailStorage()

# # ============================================================================
# # TOOL 1: SEND EMAIL
# # ============================================================================

# @mcp.tool(
#     name="send_email",
#     description="Sends a new email immediately. Supports plain text, HTML, or multipart emails with optional file attachments."
# )
# async def send_email(
#     to: List[str] = Field(description="List of recipient email addresses"),
#     subject: str = Field(description="Email subject line"),
#     body: str = Field(description="Email body content (plain text or HTML based on mimeType)"),
#     cc: Optional[List[str]] = Field(default=None, description="List of CC email addresses"),
#     bcc: Optional[List[str]] = Field(default=None, description="List of BCC email addresses"),
#     mimeType: str = Field(default="text/plain", description="MIME type of the email: text/plain, text/html, or multipart/alternative"),
#     htmlBody: Optional[str] = Field(default=None, description="HTML body for multipart emails"),
#     attachments: Optional[List[str]] = Field(default=None, description="List of file paths to attach"),
#     sender_email: Optional[str] = Field(default=None, description="Sender's email address (auto-detected if not provided)"),
#     importance: Optional[str] = Field(default="Normal", description="Email importance level: Normal, High, or Low")  # ADD THIS LINE
# ) -> Dict[str, Any]:
#     """Send email tool"""
#     await asyncio.sleep(0.1)
    
#     emails = storage.load_json(storage.emails_path)
    
#     message_id = storage.generate_message_id()
#     thread_id = storage.generate_thread_id()
    
#     # Process attachments
#     processed_attachments = []
#     if attachments:
#         for file_path in attachments:
#             if os.path.exists(file_path):
#                 file_size = os.path.getsize(file_path)
#                 file_name = os.path.basename(file_path)
#                 attachment_id = storage.generate_attachment_id()
                
#                 processed_attachments.append({
#                     "attachment_id": attachment_id,
#                     "filename": file_name,
#                     "file_path": file_path,
#                     "size_bytes": file_size,
#                     "mime_type": "application/octet-stream"
#                 })
    
#     new_email = {
#         "message_id": message_id,
#         "thread_id": thread_id,
#         "date": storage.get_current_datetime(),
#         "from": sender_email or "system@company.com",
#         "to": to,
#         "cc": cc or [],
#         "bcc": bcc or [],
#         "subject": subject,
#         "body": body,
#         "html_body": htmlBody,
#         "mime_type": mimeType,
#         "status": "sent",
#         "labels": ["SENT", "INBOX"],  # Add to both sender's SENT and recipient's INBOX
#         "attachments": processed_attachments,
#         "importance": importance or "Normal"  # ADD THIS LINE
#     }
    
#     emails.append(new_email)
#     storage.save_json(storage.emails_path, emails)
    
#     return {
#         "status": "success",
#         "message_id": message_id,
#         "thread_id": thread_id,
#         "sent_to": to,
#         "attachments_count": len(processed_attachments)
#     }


# # ============================================================================
# # TOOL 2: DRAFT EMAIL
# # ============================================================================

# @mcp.tool(
#     name="draft_email",
#     description="Creates a draft email without sending it. Also supports attachments."
# )
# async def draft_email(
#     to: List[str] = Field(description="List of recipient email addresses"),
#     subject: str = Field(description="Email subject line"),
#     body: str = Field(description="Email body content"),
#     cc: Optional[List[str]] = Field(default=None, description="List of CC email addresses"),
#     attachments: Optional[List[str]] = Field(default=None, description="List of file paths to attach"),
#     sender_email: Optional[str] = Field(default=None, description="Sender's email address")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: draft_email
#     Description: Creates a draft email without sending
#     Required Args: to, subject, body
#     Optional Args: cc, attachments, sender_email
#     """
#     emails = storage.load_json(storage.emails_path)
    
#     message_id = storage.generate_message_id()
#     thread_id = storage.generate_thread_id()
    
#     # Process attachments
#     processed_attachments = []
#     if attachments:
#         for file_path in attachments:
#             if os.path.exists(file_path):
#                 file_size = os.path.getsize(file_path)
#                 file_name = os.path.basename(file_path)
#                 attachment_id = storage.generate_attachment_id()
                
#                 processed_attachments.append({
#                     "attachment_id": attachment_id,
#                     "filename": file_name,
#                     "file_path": file_path,
#                     "size_bytes": file_size,
#                     "mime_type": "application/octet-stream"
#                 })
    
#     draft = {
#         "message_id": message_id,
#         "thread_id": thread_id,
#         "date": storage.get_current_datetime(),
#         "from": sender_email or "system@company.com",
#         "to": to,
#         "cc": cc or [],
#         "bcc": [],
#         "subject": subject,
#         "body": body,
#         "html_body": None,
#         "mime_type": "text/plain",
#         "status": "draft",
#         "labels": ["DRAFT"],
#         "attachments": processed_attachments,
#         "importance": "Normal"
#     }
    
#     emails.append(draft)
#     storage.save_json(storage.emails_path, emails)
    
#     return {
#         "status": "success",
#         "message_id": message_id,
#         "thread_id": thread_id,
#         "draft_to": to,
#         "attachments_count": len(processed_attachments)
#     }


# # ============================================================================
# # TOOL 3: READ EMAIL
# # ============================================================================

# @mcp.tool(
#     name="read_email",
#     description="Retrieves the content of a specific email by its ID. Shows enhanced attachment information."
# )
# async def read_email(
#     messageId: str = Field(description="The unique message ID of the email to read")
# ) -> str:
#     """
#     Tool Name: read_email
#     Description: Reads a specific email by ID
#     Required Args: messageId
#     """
#     emails = storage.load_json(storage.emails_path)
    
#     for email in emails:
#         if email.get("message_id") == messageId:
#             output = f"""Subject: {email.get('subject')}
# From: {email.get('from')}
# To: {', '.join(email.get('to', []))}
# CC: {', '.join(email.get('cc', [])) if email.get('cc') else 'None'}
# Date: {email.get('date')}
# Status: {email.get('status')}
# Labels: {', '.join(email.get('labels', []))}

# {email.get('body')}
# """
            
#             # Add attachment information
#             attachments = email.get('attachments', [])
#             if attachments:
#                 output += f"\n\nAttachments ({len(attachments)}):\n"
#                 for att in attachments:
#                     size_kb = att.get('size_bytes', 0) // 1024
#                     output += f"- {att.get('filename')} ({att.get('mime_type')}, {size_kb} KB, ID: {att.get('attachment_id')})\n"
            
#             return output
    
#     return f"Email with message ID {messageId} not found"


# # ============================================================================
# # TOOL 4: DOWNLOAD ATTACHMENT
# # ============================================================================

# @mcp.tool(
#     name="download_attachment",
#     description="Downloads email attachments to your local filesystem."
# )
# async def download_attachment(
#     messageId: str = Field(description="The ID of the email containing the attachment"),
#     attachmentId: str = Field(description="The attachment ID (shown in enhanced email display)"),
#     savePath: Optional[str] = Field(default=".", description="Directory to save the file (defaults to current directory)"),
#     filename: Optional[str] = Field(default=None, description="Custom filename (uses original filename if not provided)")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: download_attachment
#     Description: Downloads an email attachment
#     Required Args: messageId, attachmentId
#     Optional Args: savePath, filename
#     """
#     emails = storage.load_json(storage.emails_path)
    
#     for email in emails:
#         if email.get("message_id") == messageId:
#             attachments = email.get('attachments', [])
#             for att in attachments:
#                 if att.get('attachment_id') == attachmentId:
#                     original_path = att.get('file_path')
#                     original_filename = att.get('filename')
                    
#                     # Determine destination
#                     dest_filename = filename or original_filename
#                     dest_path = os.path.join(savePath, dest_filename)
                    
#                     # Copy file
#                     if os.path.exists(original_path):
#                         with open(original_path, 'rb') as src:
#                             with open(dest_path, 'wb') as dst:
#                                 dst.write(src.read())
                        
#                         return {
#                             "status": "success",
#                             "saved_to": dest_path,
#                             "filename": dest_filename,
#                             "size_bytes": att.get('size_bytes')
#                         }
#                     else:
#                         return {
#                             "status": "error",
#                             "message": f"Source file not found: {original_path}"
#                         }
            
#             return {
#                 "status": "error",
#                 "message": f"Attachment {attachmentId} not found in message {messageId}"
#             }
    
#     return {
#         "status": "error",
#         "message": f"Message {messageId} not found"
#     }


# # ============================================================================
# # TOOL 5: SEARCH EMAILS
# # ============================================================================

# @mcp.tool(
#     name="search_emails",
#     description="Searches for emails using Gmail search syntax (e.g., 'from:sender@example.com after:2024/01/01 has:attachment')."
# )
# async def search_emails(
#     query: str = Field(description="Gmail-style search query string"),
#     maxResults: int = Field(default=10, description="Maximum number of results to return")
# ) -> List[Dict[str, Any]]:
#     """
#     Search emails with Gmail-like syntax
#     Supports: from:, to:, subject:, has:attachment, INBOX, SENT, STARRED, etc.
#     """
#     await asyncio.sleep(0.1)
    
#     emails = storage.load_json(storage.emails_path)
#     results = []
    
#     # Parse query (simplified version)
#     query_lower = query.lower()
    
#     logger.info(f"🔍 Searching emails with query: '{query}'")
#     logger.info(f"📊 Total emails in storage: {len(emails)}")
    
#     for email in emails:
#         match = True
        
#         # Check "from:" filter
#         if "from:" in query_lower:
#             from_val = query_lower.split("from:")[1].split()[0]
#             if from_val not in (email.get("from", "") or email.get("sender_email", "")).lower():
#                 match = False
        
#         # Check "to:" filter
#         if "to:" in query_lower:
#             to_val = query_lower.split("to:")[1].split()[0]
#             to_emails = email.get("to", []) if isinstance(email.get("to"), list) else [email.get("to", "")]
#             to_emails.append(email.get("recipient_email", ""))
#             if not any(to_val in str(recipient).lower() for recipient in to_emails):
#                 match = False
        
#         # Check "subject:" filter
#         if "subject:" in query_lower:
#             subject_val = query_lower.split("subject:")[1].split()[0]
#             if subject_val not in email.get("subject", "").lower():
#                 match = False
        
#         # Check "has:attachment" filter
#         if "has:attachment" in query_lower:
#             if not email.get("attachments"):
#                 match = False
        
#         # Check label filters (INBOX, SENT, STARRED, etc.)
#         if "inbox" in query_lower:
#             if "INBOX" not in email.get("labels", []):
#                 match = False
        
#         if "sent" in query_lower:
#             if "SENT" not in email.get("labels", []):
#                 match = False
        
#         if "starred" in query_lower:
#             if not email.get("starred") and "STARRED" not in email.get("labels", []):
#                 match = False
        
#         if "draft" in query_lower:
#             if email.get("status") != "draft" and "DRAFT" not in email.get("labels", []):
#                 match = False
        
#         # General text search (if no specific filters)
#         if match and not any(x in query_lower for x in ["from:", "to:", "subject:", "has:", "inbox", "sent", "starred", "draft"]):
#             search_text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
#             if query_lower not in search_text:
#                 match = False
        
#         if match:
#             results.append({
#                 "message_id": email.get("message_id") or email.get("email_id"),
#                 "thread_id": email.get("thread_id"),
#                 "from": email.get("from") or email.get("sender_email"),
#                 "to": email.get("to") or email.get("recipient_email"),
#                 "subject": email.get("subject"),
#                 "body": email.get("body"),
#                 "date": email.get("date"),
#                 "labels": email.get("labels", []),
#                 "starred": email.get("starred", False),
#                 "status": email.get("status", "sent"),
#                 "has_attachments": len(email.get("attachments", [])) > 0
#             })
        
#         if len(results) >= maxResults:
#             break
    
#     logger.info(f"✅ Found {len(results)} matching emails")
#     return results


# # ============================================================================
# # TOOL 6: MODIFY EMAIL (Labels)
# # ============================================================================

# @mcp.tool(
#     name="modify_email",
#     description="Adds or removes labels from emails (move to different folders, archive, etc.)."
# )
# async def modify_email(
#     messageId: str = Field(description="The message ID to modify"),
#     addLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to add"),
#     removeLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to remove")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: modify_email
#     Description: Modifies email labels
#     Required Args: messageId
#     Optional Args: addLabelIds, removeLabelIds
#     """
#     emails = storage.load_json(storage.emails_path)
    
#     for email in emails:
#         if email.get("message_id") == messageId:
#             current_labels = set(email.get("labels", []))
            
#             if addLabelIds:
#                 current_labels.update(addLabelIds)
            
#             if removeLabelIds:
#                 current_labels.difference_update(removeLabelIds)
            
#             email["labels"] = list(current_labels)
#             storage.save_json(storage.emails_path, emails)
            
#             return {
#                 "status": "success",
#                 "message_id": messageId,
#                 "labels": email["labels"]
#             }
    
#     return {
#         "status": "error",
#         "message": f"Message {messageId} not found"
#     }


# # ============================================================================
# # TOOL 7: DELETE EMAIL
# # ============================================================================

# @mcp.tool(
#     name="delete_email",
#     description="Permanently deletes an email."
# )
# async def delete_email(
#     messageId: str = Field(description="The message ID to delete")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: delete_email
#     Description: Permanently deletes an email
#     Required Args: messageId
#     """
#     emails = storage.load_json(storage.emails_path)
    
#     for i, email in enumerate(emails):
#         if email.get("message_id") == messageId:
#             deleted = emails.pop(i)
#             storage.save_json(storage.emails_path, emails)
            
#             return {
#                 "status": "success",
#                 "message": f"Email {messageId} deleted",
#                 "subject": deleted.get("subject")
#             }
    
#     return {
#         "status": "error",
#         "message": f"Message {messageId} not found"
#     }


# # ============================================================================
# # TOOL 8: LIST EMAIL LABELS
# # ============================================================================

# @mcp.tool(
#     name="list_email_labels",
#     description="Retrieves all available Gmail labels."
# )
# async def list_email_labels() -> List[Dict[str, Any]]:
#     """
#     Tool Name: list_email_labels
#     Description: Lists all email labels
#     Required Args: None
#     """
#     # Standard Gmail labels
#     labels = [
#         {"id": "INBOX", "name": "INBOX", "type": "system"},
#         {"id": "SENT", "name": "SENT", "type": "system"},
#         {"id": "DRAFT", "name": "DRAFT", "type": "system"},
#         {"id": "TRASH", "name": "TRASH", "type": "system"},
#         {"id": "SPAM", "name": "SPAM", "type": "system"},
#         {"id": "IMPORTANT", "name": "IMPORTANT", "type": "system"},
#         {"id": "STARRED", "name": "STARRED", "type": "system"},
#     ]
    
#     # Load custom labels from emails (extract unique labels)
#     emails = storage.load_json(storage.emails_path)
#     custom_labels = set()
    
#     for email in emails:
#         for label in email.get("labels", []):
#             if label not in ["INBOX", "SENT", "DRAFT", "TRASH", "SPAM", "IMPORTANT", "STARRED"]:
#                 custom_labels.add(label)
    
#     for label in custom_labels:
#         labels.append({
#             "id": label,
#             "name": label,
#             "type": "user"
#         })
    
#     return labels


# # ============================================================================
# # TOOL 9: CREATE LABEL
# # ============================================================================

# @mcp.tool(
#     name="create_label",
#     description="Creates a new Gmail label."
# )
# async def create_label(
#     name: str = Field(description="Label name"),
#     messageListVisibility: Literal["show", "hide"] = Field(default="show", description="Visibility in message list"),
#     labelListVisibility: Literal["labelShow", "labelHide"] = Field(default="labelShow", description="Visibility in label list")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: create_label
#     Description: Creates a new email label
#     Required Args: name
#     Optional Args: messageListVisibility, labelListVisibility
#     """
#     label_id = storage.generate_label_id()
    
#     return {
#         "status": "success",
#         "id": label_id,
#         "name": name,
#         "messageListVisibility": messageListVisibility,
#         "labelListVisibility": labelListVisibility,
#         "type": "user"
#     }


# # ============================================================================
# # TOOL 10: UPDATE LABEL
# # ============================================================================

# @mcp.tool(
#     name="update_label",
#     description="Updates an existing Gmail label."
# )
# async def update_label(
#     id: str = Field(description="Label ID to update"),
#     name: str = Field(description="New label name"),
#     messageListVisibility: Optional[Literal["show", "hide"]] = Field(default=None, description="Visibility in message list"),
#     labelListVisibility: Optional[Literal["labelShow", "labelHide"]] = Field(default=None, description="Visibility in label list")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: update_label
#     Description: Updates an existing label
#     Required Args: id, name
#     Optional Args: messageListVisibility, labelListVisibility
#     """
#     # Update all emails with this label
#     emails = storage.load_json(storage.emails_path)
#     updated_count = 0
    
#     for email in emails:
#         labels = email.get("labels", [])
#         if id in labels:
#             # Replace old label with new name
#             labels[labels.index(id)] = name
#             email["labels"] = labels
#             updated_count += 1
    
#     if updated_count > 0:
#         storage.save_json(storage.emails_path, emails)
    
#     return {
#         "status": "success",
#         "id": id,
#         "name": name,
#         "emails_updated": updated_count
#     }


# # ============================================================================
# # TOOL 11: DELETE LABEL
# # ============================================================================

# @mcp.tool(
#     name="delete_label",
#     description="Deletes a Gmail label."
# )
# async def delete_label(
#     id: str = Field(description="Label ID to delete")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: delete_label
#     Description: Deletes a label
#     Required Args: id
#     """
#     # Remove label from all emails
#     emails = storage.load_json(storage.emails_path)
#     removed_count = 0
    
#     for email in emails:
#         labels = email.get("labels", [])
#         if id in labels:
#             labels.remove(id)
#             email["labels"] = labels
#             removed_count += 1
    
#     if removed_count > 0:
#         storage.save_json(storage.emails_path, emails)
    
#     return {
#         "status": "success",
#         "id": id,
#         "removed_from_emails": removed_count
#     }


# # ============================================================================
# # TOOL 12: GET OR CREATE LABEL
# # ============================================================================

# @mcp.tool(
#     name="get_or_create_label",
#     description="Gets an existing label by name or creates it if it doesn't exist."
# )
# async def get_or_create_label(
#     name: str = Field(description="Label name"),
#     messageListVisibility: Literal["show", "hide"] = Field(default="show", description="Visibility in message list"),
#     labelListVisibility: Literal["labelShow", "labelHide"] = Field(default="labelShow", description="Visibility in label list")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: get_or_create_label
#     Description: Gets or creates a label
#     Required Args: name
#     Optional Args: messageListVisibility, labelListVisibility
#     """
#     # Check if label exists
#     emails = storage.load_json(storage.emails_path)
    
#     for email in emails:
#         if name in email.get("labels", []):
#             return {
#                 "status": "exists",
#                 "id": name,
#                 "name": name,
#                 "type": "user"
#             }
    
#     # Create new label
#     label_id = storage.generate_label_id()
    
#     return {
#         "status": "created",
#         "id": label_id,
#         "name": name,
#         "messageListVisibility": messageListVisibility,
#         "labelListVisibility": labelListVisibility,
#         "type": "user"
#     }


# # ============================================================================
# # TOOL 13: BATCH MODIFY EMAILS
# # ============================================================================

# @mcp.tool(
#     name="batch_modify_emails",
#     description="Modifies labels for multiple emails in efficient batches."
# )
# async def batch_modify_emails(
#     messageIds: List[str] = Field(description="List of message IDs to modify"),
#     addLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to add"),
#     removeLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to remove"),
#     batchSize: int = Field(default=50, description="Batch size for processing")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: batch_modify_emails
#     Description: Batch modifies email labels
#     Required Args: messageIds
#     Optional Args: addLabelIds, removeLabelIds, batchSize
#     """
#     emails = storage.load_json(storage.emails_path)
#     modified_count = 0
    
#     for email in emails:
#         if email.get("message_id") in messageIds:
#             current_labels = set(email.get("labels", []))
            
#             if addLabelIds:
#                 current_labels.update(addLabelIds)
            
#             if removeLabelIds:
#                 current_labels.difference_update(removeLabelIds)
            
#             email["labels"] = list(current_labels)
#             modified_count += 1
    
#     if modified_count > 0:
#         storage.save_json(storage.emails_path, emails)
    
#     return {
#         "status": "success",
#         "modified_count": modified_count,
#         "total_requested": len(messageIds)
#     }


# # ============================================================================
# # TOOL 14: BATCH DELETE EMAILS
# # ============================================================================

# @mcp.tool(
#     name="batch_delete_emails",
#     description="Permanently deletes multiple emails in efficient batches."
# )
# async def batch_delete_emails(
#     messageIds: List[str] = Field(description="List of message IDs to delete"),
#     batchSize: int = Field(default=50, description="Batch size for processing")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: batch_delete_emails
#     Description: Batch deletes emails
#     Required Args: messageIds
#     Optional Args: batchSize
#     """
#     emails = storage.load_json(storage.emails_path)
#     initial_count = len(emails)
    
#     # Filter out emails to delete
#     emails = [email for email in emails if email.get("message_id") not in messageIds]
    
#     deleted_count = initial_count - len(emails)
    
#     if deleted_count > 0:
#         storage.save_json(storage.emails_path, emails)
    
#     return {
#         "status": "success",
#         "deleted_count": deleted_count,
#         "total_requested": len(messageIds)
#     }


# # ============================================================================
# # TOOL 15: CREATE FILTER
# # ============================================================================

# @mcp.tool(
#     name="create_filter",
#     description="Creates a new Gmail filter with custom criteria and actions."
# )
# async def create_filter(
#     criteria: Dict[str, Any] = Field(description="Filter criteria (from, to, subject, query, hasAttachment, etc.)"),
#     action: Dict[str, Any] = Field(description="Actions to take (addLabelIds, removeLabelIds, forward, etc.)")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: create_filter
#     Description: Creates an email filter
#     Required Args: criteria, action
#     """
#     filter_id = storage.generate_filter_id()
    
#     # Store filter (in a real implementation, this would be in a separate filters.json)
#     filter_obj = {
#         "id": filter_id,
#         "criteria": criteria,
#         "action": action,
#         "created_date": storage.get_current_datetime()
#     }
    
#     return {
#         "status": "success",
#         "filter_id": filter_id,
#         "criteria": criteria,
#         "action": action
#     }


# # ============================================================================
# # TOOL 16: LIST FILTERS
# # ============================================================================

# @mcp.tool(
#     name="list_filters",
#     description="Retrieves all Gmail filters."
# )
# async def list_filters() -> List[Dict[str, Any]]:
#     """
#     Tool Name: list_filters
#     Description: Lists all email filters
#     Required Args: None
#     """
#     # In a real implementation, load from filters.json
#     return []


# # ============================================================================
# # TOOL 17: GET FILTER
# # ============================================================================

# @mcp.tool(
#     name="get_filter",
#     description="Gets details of a specific Gmail filter."
# )
# async def get_filter(
#     filterId: str = Field(description="Filter ID to retrieve")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: get_filter
#     Description: Gets a specific filter
#     Required Args: filterId
#     """
#     # In a real implementation, load from filters.json
#     return {
#         "status": "error",
#         "message": f"Filter {filterId} not found"
#     }


# # ============================================================================
# # TOOL 18: DELETE FILTER
# # ============================================================================

# @mcp.tool(
#     name="delete_filter",
#     description="Deletes a Gmail filter."
# )
# async def delete_filter(
#     filterId: str = Field(description="Filter ID to delete")
# ) -> Dict[str, Any]:
#     """
#     Tool Name: delete_filter
#     Description: Deletes a filter
#     Required Args: filterId
#     """
#     # In a real implementation, remove from filters.json
#     return {
#         "status": "success",
#         "filter_id": filterId,
#         "message": "Filter deleted"
#     }


# # ============================================================================
# # TOOL 19: CREATE FILTER FROM TEMPLATE
# # ============================================================================

# @mcp.tool(
#     name="create_filter_from_template",
#     description="Creates a filter using pre-defined templates for common scenarios."
# )
# async def create_filter_from_template(
#     template: Literal["fromSender", "toRecipient", "hasAttachment", "labelAndArchive"] = Field(
#         description="Template name: fromSender, toRecipient, hasAttachment, labelAndArchive"
#     ),
#     parameters: Dict[str, Any] = Field(
#         description="Template parameters (e.g., senderEmail, labelIds, archive)"
#     )
# ) -> Dict[str, Any]:
#     """
#     Tool Name: create_filter_from_template
#     Description: Creates a filter from a template
#     Required Args: template, parameters
#     """
#     filter_id = storage.generate_filter_id()
    
#     # Build criteria and action based on template
#     if template == "fromSender":
#         criteria = {"from": parameters.get("senderEmail")}
#         action = {
#             "addLabelIds": parameters.get("labelIds", []),
#             "removeLabelIds": ["INBOX"] if parameters.get("archive") else []
#         }
#     elif template == "hasAttachment":
#         criteria = {"hasAttachment": True}
#         action = {"addLabelIds": parameters.get("labelIds", [])}
#     else:
#         criteria = parameters.get("criteria", {})
#         action = parameters.get("action", {})
    
#     return {
#         "status": "success",
#         "filter_id": filter_id,
#         "template": template,
#         "criteria": criteria,
#         "action": action
#     }


# # ============================================================================
# # RUN SERVER
# # ============================================================================

# if __name__ == "__main__":
#     # Run the MCP server
#     mcp.run(transport="stdio")


# ============================================================================
# ENTERPRISE EMAIL MCP SERVER - HYBRID APPROACH
# Storage + Background API Sync
# ============================================================================

import asyncio
from typing import List, Dict, Any, Optional, Literal
from fastmcp import FastMCP
from pydantic import Field
from EnterpriseLab.Arena.MCP_servers.email.email_storage import EmailStorage
from aiohttp import ClientSession
import base64
import os
import logging


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# MCP SERVER INITIALIZATION
# ============================================================================

mcp = FastMCP(name="Enterprise Email MCP Server")


# ============================================================================
# STORAGE & CLIENT INITIALIZATION
# ============================================================================

storage = EmailStorage()


class EmailAPIPClient:
    """HTTP Client for background API sync"""
    
    def __init__(self, base_url: str = "http://localhost:12005"):
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
        Make API request to Email Backend
        
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


email_api_client = EmailAPIPClient(base_url="http://localhost:12005")


# ============================================================================
# TOOL 1: SEND EMAIL - HYBRID
# ============================================================================

@mcp.tool(
    name="send_email",
    description="Sends a new email immediately. Supports plain text, HTML, or multipart emails with optional file attachments."
)
async def send_email(
    to: List[str] = Field(description="List of recipient email addresses"),
    subject: str = Field(description="Email subject line"),
    body: str = Field(description="Email body content (plain text or HTML based on mimeType)"),
    cc: Optional[List[str]] = Field(default=None, description="List of CC email addresses"),
    bcc: Optional[List[str]] = Field(default=None, description="List of BCC email addresses"),
    mimeType: str = Field(default="text/plain", description="MIME type: text/plain, text/html, or multipart/alternative"),
    htmlBody: Optional[str] = Field(default=None, description="HTML body for multipart emails"),
    attachments: Optional[List[str]] = Field(default=None, description="List of file paths to attach"),
    sender_email: Optional[str] = Field(default=None, description="Sender's email address (auto-detected if not provided)"),
    importance: Optional[str] = Field(default="Normal", description="Email importance: Normal, High, or Low")
) -> Dict[str, Any]:
    """Send email"""
    
    # STEP 1: Local storage operations
    await asyncio.sleep(0.1)
    
    emails = storage.load_json(storage.emails_path)
    
    message_id = storage.generate_message_id()
    thread_id = storage.generate_thread_id()
    
    # Process attachments
    processed_attachments = []
    if attachments:
        for file_path in attachments:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                attachment_id = storage.generate_attachment_id()
                
                processed_attachments.append({
                    "attachment_id": attachment_id,
                    "filename": file_name,
                    "file_path": file_path,
                    "size_bytes": file_size,
                    "mime_type": "application/octet-stream"
                })
    
    new_email = {
        "message_id": message_id,
        "thread_id": thread_id,
        "date": storage.get_current_datetime(),
        "from": sender_email or "system@company.com",
        "to": to,
        "cc": cc or [],
        "bcc": bcc or [],
        "subject": subject,
        "body": body,
        "html_body": htmlBody,
        "mime_type": mimeType,
        "status": "sent",
        "labels": ["SENT", "INBOX"],
        "attachments": processed_attachments,
        "importance": importance or "Normal"
    }
    
    emails.append(new_email)
    storage.save_json(storage.emails_path, emails)
    
    # STEP 2: Background API sync (fire-and-forget)
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="emails/send",
            method="POST",
            data={
                "to": ", ".join(to),
                "subject": subject,
                "body": body,
                "cc": ", ".join(cc) if cc else "",
                "sender_email": sender_email,
                "importance": importance
            }
        )
    )
    
    logger.info(f"✅ Email sent: {message_id} to {to}")
    return {
        "status": "success",
        "message_id": message_id,
        "thread_id": thread_id,
        "sent_to": to,
        "attachments_count": len(processed_attachments)
    }


# ============================================================================
# TOOL 2: DRAFT EMAIL - HYBRID
# ============================================================================

@mcp.tool(
    name="draft_email",
    description="Creates a draft email without sending it. Also supports attachments."
)
async def draft_email(
    to: List[str] = Field(description="List of recipient email addresses"),
    subject: str = Field(description="Email subject line"),
    body: str = Field(description="Email body content"),
    cc: Optional[List[str]] = Field(default=None, description="List of CC email addresses"),
    attachments: Optional[List[str]] = Field(default=None, description="List of file paths to attach"),
    sender_email: Optional[str] = Field(default=None, description="Sender's email address")
) -> Dict[str, Any]:
    """Draft email"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    
    message_id = storage.generate_message_id()
    thread_id = storage.generate_thread_id()
    
    # Process attachments
    processed_attachments = []
    if attachments:
        for file_path in attachments:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                attachment_id = storage.generate_attachment_id()
                
                processed_attachments.append({
                    "attachment_id": attachment_id,
                    "filename": file_name,
                    "file_path": file_path,
                    "size_bytes": file_size,
                    "mime_type": "application/octet-stream"
                })
    
    draft = {
        "message_id": message_id,
        "thread_id": thread_id,
        "date": storage.get_current_datetime(),
        "from": sender_email or "system@company.com",
        "to": to,
        "cc": cc or [],
        "bcc": [],
        "subject": subject,
        "body": body,
        "html_body": None,
        "mime_type": "text/plain",
        "status": "draft",
        "labels": ["DRAFT"],
        "attachments": processed_attachments,
        "importance": "Normal"
    }
    
    emails.append(draft)
    storage.save_json(storage.emails_path, emails)
    
    # STEP 2: Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="emails/draft",
            method="POST",
            data={
                "to": ", ".join(to),
                "subject": subject,
                "body": body,
                "cc": ", ".join(cc) if cc else "",
                "sender_email": sender_email
            }
        )
    )
    
    logger.info(f"✅ Draft created: {message_id}")
    return {
        "status": "success",
        "message_id": message_id,
        "thread_id": thread_id,
        "draft_to": to,
        "attachments_count": len(processed_attachments)
    }


# ============================================================================
# TOOL 3: READ EMAIL
# ============================================================================

@mcp.tool(
    name="read_email",
    description="Retrieves the content of a specific email by its ID. Shows enhanced attachment information."
)
async def read_email(
    messageId: str = Field(description="The unique message ID of the email to read")
) -> str:
    """Read email"""
    emails = storage.load_json(storage.emails_path)
    
    for email in emails:
        if email.get("message_id") == messageId:
            output = f"""Subject: {email.get('subject')}
From: {email.get('from')}
To: {', '.join(email.get('to', []))}
CC: {', '.join(email.get('cc', [])) if email.get('cc') else 'None'}
Date: {email.get('date')}
Status: {email.get('status')}
Labels: {', '.join(email.get('labels', []))}


{email.get('body')}
"""
            
            # Add attachment information
            attachments = email.get('attachments', [])
            if attachments:
                output += f"\\n\\nAttachments ({len(attachments)}):\\n"
                for att in attachments:
                    size_kb = att.get('size_bytes', 0) // 1024
                    output += f"- {att.get('filename')} ({att.get('mime_type')}, {size_kb} KB, ID: {att.get('attachment_id')})\\n"
            
            return output
    
    return f"Email with message ID {messageId} not found"


# ============================================================================
# TOOL 4: DOWNLOAD ATTACHMENT - HYBRID
# ============================================================================

@mcp.tool(
    name="download_attachment",
    description="Downloads email attachments to your local filesystem."
)
async def download_attachment(
    messageId: str = Field(description="The ID of the email containing the attachment"),
    attachmentId: str = Field(description="The attachment ID"),
    savePath: Optional[str] = Field(default=".", description="Directory to save the file"),
    filename: Optional[str] = Field(default=None, description="Custom filename")
) -> Dict[str, Any]:
    """Download attachment"""
    emails = storage.load_json(storage.emails_path)
    
    for email in emails:
        if email.get("message_id") == messageId:
            attachments = email.get('attachments', [])
            for att in attachments:
                if att.get('attachment_id') == attachmentId:
                    original_path = att.get('file_path')
                    original_filename = att.get('filename')
                    
                    # Determine destination
                    dest_filename = filename or original_filename
                    dest_path = os.path.join(savePath, dest_filename)
                    
                    # Copy file
                    if os.path.exists(original_path):
                        with open(original_path, 'rb') as src:
                            with open(dest_path, 'wb') as dst:
                                dst.write(src.read())
                        
                        # Background API sync
                        asyncio.create_task(
                            email_api_client.api_request(
                                endpoint="attachments/download",
                                method="POST",
                                data={
                                    "messageId": messageId,
                                    "attachmentId": attachmentId,
                                    "savePath": savePath,
                                    "filename": filename
                                }
                            )
                        )
                        
                        return {
                            "status": "success",
                            "saved_to": dest_path,
                            "filename": dest_filename,
                            "size_bytes": att.get('size_bytes')
                        }
                    else:
                        return {"status": "error", "message": f"Source file not found: {original_path}"}
            
            return {"status": "error", "message": f"Attachment {attachmentId} not found"}
    
    return {"status": "error", "message": f"Message {messageId} not found"}


# ============================================================================
# TOOL 5: SEARCH EMAILS - HYBRID
# ============================================================================

@mcp.tool(
    name="search_emails",
    description="Searches for emails using Gmail search syntax."
)
async def search_emails(
    query: str = Field(description="Gmail-style search query string"),
    maxResults: int = Field(default=10, description="Maximum number of results")
) -> List[Dict[str, Any]]:
    """Search emails"""
    
    await asyncio.sleep(0.1)
    
    # STEP 1: Local storage search
    emails = storage.load_json(storage.emails_path)
    results = []
    
    logger.info(f"🔍 Searching emails with query: '{query}'")
    logger.info(f"📊 Total emails in storage: {len(emails)}")
    
    query_lower = query.lower()
    
    for email in emails:
        match = True
        
        # Check "from:" filter
        if "from:" in query_lower:
            from_val = query_lower.split("from:")[1].split()[0]
            if from_val not in (email.get("from", "") or email.get("sender_email", "")).lower():
                match = False
        
        # Check "to:" filter
        if "to:" in query_lower:
            to_val = query_lower.split("to:")[1].split()[0]
            to_emails = email.get("to", []) if isinstance(email.get("to"), list) else [email.get("to", "")]
            to_emails.append(email.get("recipient_email", ""))
            if not any(to_val in str(recipient).lower() for recipient in to_emails):
                match = False
        
        # Check "subject:" filter
        if "subject:" in query_lower:
            subject_val = query_lower.split("subject:")[1].split()[0]
            if subject_val not in email.get("subject", "").lower():
                match = False
        
        # Check "has:attachment" filter
        if "has:attachment" in query_lower:
            if not email.get("attachments"):
                match = False
        
        # Check label filters
        if "inbox" in query_lower:
            if "INBOX" not in email.get("labels", []):
                match = False
        
        if "sent" in query_lower:
            if "SENT" not in email.get("labels", []):
                match = False
        
        if "starred" in query_lower:
            if not email.get("starred") and "STARRED" not in email.get("labels", []):
                match = False
        
        if "draft" in query_lower:
            if email.get("status") != "draft" and "DRAFT" not in email.get("labels", []):
                match = False
        
        # General text search
        if match and not any(x in query_lower for x in ["from:", "to:", "subject:", "has:", "inbox", "sent", "starred", "draft"]):
            search_text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
            if query_lower not in search_text:
                match = False
        
        if match:
            results.append({
                "message_id": email.get("message_id") or email.get("email_id"),
                "thread_id": email.get("thread_id"),
                "from": email.get("from") or email.get("sender_email"),
                "to": email.get("to") or email.get("recipient_email"),
                "subject": email.get("subject"),
                "body": email.get("body"),
                "date": email.get("date"),
                "labels": email.get("labels", []),
                "starred": email.get("starred", False),
                "status": email.get("status", "sent"),
                "has_attachments": len(email.get("attachments", [])) > 0
            })
        
        if len(results) >= maxResults:
            break
    
    logger.info(f"✅ Found {len(results)} matching emails")
    
    # STEP 2: Background API sync (fire-and-forget)
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="emails/search",
            method="POST",
            data={
                "query": query,
                "max_results": maxResults
            }
        )
    )
    
    return results


# ============================================================================
# TOOL 6: MODIFY EMAIL (Labels) - HYBRID
# ============================================================================

@mcp.tool(
    name="modify_email",
    description="Adds or removes labels from emails."
)
async def modify_email(
    messageId: str = Field(description="The message ID to modify"),
    addLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to add"),
    removeLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to remove")
) -> Dict[str, Any]:
    """Modify email"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    
    for email in emails:
        if email.get("message_id") == messageId:
            current_labels = set(email.get("labels", []))
            
            if addLabelIds:
                current_labels.update(addLabelIds)
            
            if removeLabelIds:
                current_labels.difference_update(removeLabelIds)
            
            email["labels"] = list(current_labels)
            storage.save_json(storage.emails_path, emails)
            
            # STEP 2: Background API sync
            asyncio.create_task(
                email_api_client.api_request(
                    endpoint=f"emails/{messageId}/labels",
                    method="PUT",
                    data={
                        "message_id": messageId,
                        "add_labels": addLabelIds,
                        "remove_labels": removeLabelIds
                    }
                )
            )
            
            logger.info(f"✏️ Email labels modified: {messageId}")
            return {
                "status": "success",
                "message_id": messageId,
                "labels": email["labels"]
            }
    
    return {"status": "error", "message": f"Message {messageId} not found"}


# ============================================================================
# TOOL 7: DELETE EMAIL - HYBRID
# ============================================================================

@mcp.tool(
    name="delete_email",
    description="Permanently deletes an email."
)
async def delete_email(
    messageId: str = Field(description="The message ID to delete")
) -> Dict[str, Any]:
    """Delete email"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    
    for i, email in enumerate(emails):
        if email.get("message_id") == messageId:
            deleted = emails.pop(i)
            storage.save_json(storage.emails_path, emails)
            
            # STEP 2: Background API sync
            asyncio.create_task(
                email_api_client.api_request(
                    endpoint=f"emails/{messageId}",
                    method="DELETE"
                )
            )
            
            logger.info(f"🗑️ Email deleted: {messageId}")
            return {
                "status": "success",
                "message": f"Email {messageId} deleted",
                "subject": deleted.get("subject")
            }
    
    return {"status": "error", "message": f"Message {messageId} not found"}


# ============================================================================
# TOOL 8: LIST EMAIL LABELS
# ============================================================================

@mcp.tool(
    name="list_email_labels",
    description="Retrieves all available Gmail labels."
)
async def list_email_labels() -> List[Dict[str, Any]]:
    """List labels"""
    
    # Standard Gmail labels
    labels = [
        {"id": "INBOX", "name": "INBOX", "type": "system"},
        {"id": "SENT", "name": "SENT", "type": "system"},
        {"id": "DRAFT", "name": "DRAFT", "type": "system"},
        {"id": "TRASH", "name": "TRASH", "type": "system"},
        {"id": "SPAM", "name": "SPAM", "type": "system"},
        {"id": "IMPORTANT", "name": "IMPORTANT", "type": "system"},
        {"id": "STARRED", "name": "STARRED", "type": "system"},
    ]
    
    # Load custom labels from emails
    emails = storage.load_json(storage.emails_path)
    custom_labels = set()
    
    for email in emails:
        for label in email.get("labels", []):
            if label not in ["INBOX", "SENT", "DRAFT", "TRASH", "SPAM", "IMPORTANT", "STARRED"]:
                custom_labels.add(label)
    
    for label in custom_labels:
        labels.append({"id": label, "name": label, "type": "user"})
    
    return labels


# ============================================================================
# TOOL 9: CREATE LABEL - HYBRID
# ============================================================================

@mcp.tool(
    name="create_label",
    description="Creates a new Gmail label."
)
async def create_label(
    name: str = Field(description="Label name"),
    messageListVisibility: Literal["show", "hide"] = Field(default="show", description="Visibility in message list"),
    labelListVisibility: Literal["labelShow", "labelHide"] = Field(default="labelShow", description="Visibility in label list")
) -> Dict[str, Any]:
    """Create label"""
    
    label_id = storage.generate_label_id()
    
    # Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="labels",
            method="POST",
            data={
                "name": name,
                "messageListVisibility": messageListVisibility,
                "labelListVisibility": labelListVisibility
            }
        )
    )
    
    logger.info(f"✅ Label created: {label_id} - {name}")
    return {
        "status": "success",
        "id": label_id,
        "name": name,
        "messageListVisibility": messageListVisibility,
        "labelListVisibility": labelListVisibility,
        "type": "user"
    }


# ============================================================================
# TOOL 10: UPDATE LABEL - HYBRID
# ============================================================================

@mcp.tool(
    name="update_label",
    description="Updates an existing Gmail label."
)
async def update_label(
    id: str = Field(description="Label ID to update"),
    name: str = Field(description="New label name"),
    messageListVisibility: Optional[Literal["show", "hide"]] = Field(default=None, description="Visibility in message list"),
    labelListVisibility: Optional[Literal["labelShow", "labelHide"]] = Field(default=None, description="Visibility in label list")
) -> Dict[str, Any]:
    """Update label"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    updated_count = 0
    
    for email in emails:
        labels = email.get("labels", [])
        if id in labels:
            labels[labels.index(id)] = name
            email["labels"] = labels
            updated_count += 1
    
    if updated_count > 0:
        storage.save_json(storage.emails_path, emails)
    
    # STEP 2: Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint=f"labels/{id}",
            method="PUT",
            data={
                "name": name,
                "messageListVisibility": messageListVisibility,
                "labelListVisibility": labelListVisibility
            }
        )
    )
    
    logger.info(f"✏️ Label updated: {id} -> {name}")
    return {
        "status": "success",
        "id": id,
        "name": name,
        "emails_updated": updated_count
    }


# ============================================================================
# TOOL 11: DELETE LABEL - HYBRID
# ============================================================================

@mcp.tool(
    name="delete_label",
    description="Deletes a Gmail label."
)
async def delete_label(
    id: str = Field(description="Label ID to delete")
) -> Dict[str, Any]:
    """Delete label"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    removed_count = 0
    
    for email in emails:
        labels = email.get("labels", [])
        if id in labels:
            labels.remove(id)
            email["labels"] = labels
            removed_count += 1
    
    if removed_count > 0:
        storage.save_json(storage.emails_path, emails)
    
    # STEP 2: Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint=f"labels/{id}",
            method="DELETE"
        )
    )
    
    logger.info(f"🗑️ Label deleted: {id}")
    return {
        "status": "success",
        "id": id,
        "removed_from_emails": removed_count
    }


# ============================================================================
# TOOL 12: GET OR CREATE LABEL - HYBRID
# ============================================================================

@mcp.tool(
    name="get_or_create_label",
    description="Gets an existing label by name or creates it if it doesn't exist."
)
async def get_or_create_label(
    name: str = Field(description="Label name"),
    messageListVisibility: Literal["show", "hide"] = Field(default="show", description="Visibility in message list"),
    labelListVisibility: Literal["labelShow", "labelHide"] = Field(default="labelShow", description="Visibility in label list")
) -> Dict[str, Any]:
    """Get or create label"""
    
    # STEP 1: Check local storage
    emails = storage.load_json(storage.emails_path)
    
    for email in emails:
        if name in email.get("labels", []):
            return {
                "status": "exists",
                "id": name,
                "name": name,
                "type": "user"
            }
    
    # Create new label
    label_id = storage.generate_label_id()
    
    # Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="labels/get-or-create",
            method="POST",
            data={
                "name": name,
                "messageListVisibility": messageListVisibility,
                "labelListVisibility": labelListVisibility
            }
        )
    )
    
    logger.info(f"✅ Label created: {label_id} - {name}")
    return {
        "status": "created",
        "id": label_id,
        "name": name,
        "messageListVisibility": messageListVisibility,
        "labelListVisibility": labelListVisibility,
        "type": "user"
    }


# ============================================================================
# TOOL 13: BATCH MODIFY EMAILS - HYBRID
# ============================================================================

@mcp.tool(
    name="batch_modify_emails",
    description="Modifies labels for multiple emails in efficient batches."
)
async def batch_modify_emails(
    messageIds: List[str] = Field(description="List of message IDs to modify"),
    addLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to add"),
    removeLabelIds: Optional[List[str]] = Field(default=None, description="List of label IDs to remove"),
    batchSize: int = Field(default=50, description="Batch size for processing")
) -> Dict[str, Any]:
    """Batch modify emails"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    modified_count = 0
    
    for email in emails:
        if email.get("message_id") in messageIds:
            current_labels = set(email.get("labels", []))
            
            if addLabelIds:
                current_labels.update(addLabelIds)
            
            if removeLabelIds:
                current_labels.difference_update(removeLabelIds)
            
            email["labels"] = list(current_labels)
            modified_count += 1
    
    if modified_count > 0:
        storage.save_json(storage.emails_path, emails)
    
    # STEP 2: Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="emails/batch-modify",
            method="POST",
            data={
                "messageIds": messageIds,
                "add_labels": addLabelIds,
                "remove_labels": removeLabelIds
            }
        )
    )
    
    logger.info(f"✏️ Batch modified {modified_count} emails")
    return {
        "status": "success",
        "modified_count": modified_count,
        "total_requested": len(messageIds)
    }


# ============================================================================
# TOOL 14: BATCH DELETE EMAILS - HYBRID
# ============================================================================

@mcp.tool(
    name="batch_delete_emails",
    description="Permanently deletes multiple emails in efficient batches."
)
async def batch_delete_emails(
    messageIds: List[str] = Field(description="List of message IDs to delete"),
    batchSize: int = Field(default=50, description="Batch size for processing")
) -> Dict[str, Any]:
    """Batch delete emails"""
    
    # STEP 1: Local storage operations
    emails = storage.load_json(storage.emails_path)
    initial_count = len(emails)
    
    # Filter out emails to delete
    emails = [email for email in emails if email.get("message_id") not in messageIds]
    
    deleted_count = initial_count - len(emails)
    
    if deleted_count > 0:
        storage.save_json(storage.emails_path, emails)
    
    # STEP 2: Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="emails/batch-delete",
            method="POST",
            data={"messageIds": messageIds}
        )
    )
    
    logger.info(f"🗑️ Batch deleted {deleted_count} emails")
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "total_requested": len(messageIds)
    }


# ============================================================================
# TOOL 15: CREATE FILTER - HYBRID
# ============================================================================

@mcp.tool(
    name="create_filter",
    description="Creates a new Gmail filter with custom criteria and actions."
)
async def create_filter(
    criteria: Dict[str, Any] = Field(description="Filter criteria"),
    action: Dict[str, Any] = Field(description="Actions to take")
) -> Dict[str, Any]:
    """Create filter"""
    
    filter_id = storage.generate_filter_id()
    
    # Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="filters",
            method="POST",
            data={
                "criteria": criteria,
                "action": action
            }
        )
    )
    
    logger.info(f"✅ Filter created: {filter_id}")
    return {
        "status": "success",
        "filter_id": filter_id,
        "criteria": criteria,
        "action": action
    }


# ============================================================================
# TOOL 16: LIST FILTERS
# ============================================================================

@mcp.tool(
    name="list_filters",
    description="Retrieves all Gmail filters."
)
async def list_filters() -> List[Dict[str, Any]]:
    """List filters"""
    
    # In a real implementation, load from filters.json
    return []


# ============================================================================
# TOOL 17: GET FILTER
# ============================================================================

@mcp.tool(
    name="get_filter",
    description="Gets details of a specific Gmail filter."
)
async def get_filter(
    filterId: str = Field(description="Filter ID to retrieve")
) -> Dict[str, Any]:
    """Get filter"""
    
    # In a real implementation, load from filters.json
    return {"status": "error", "message": f"Filter {filterId} not found"}


# ============================================================================
# TOOL 18: DELETE FILTER - HYBRID
# ============================================================================

@mcp.tool(
    name="delete_filter",
    description="Deletes a Gmail filter."
)
async def delete_filter(
    filterId: str = Field(description="Filter ID to delete")
) -> Dict[str, Any]:
    """Delete filter"""
    
    # Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint=f"filters/{filterId}",
            method="DELETE"
        )
    )
    
    logger.info(f"🗑️ Filter deleted: {filterId}")
    return {
        "status": "success",
        "filter_id": filterId,
        "message": "Filter deleted"
    }


# ============================================================================
# TOOL 19: CREATE FILTER FROM TEMPLATE - HYBRID
# ============================================================================

@mcp.tool(
    name="create_filter_from_template",
    description="Creates a filter using pre-defined templates for common scenarios."
)
async def create_filter_from_template(
    template: Literal["fromSender", "toRecipient", "hasAttachment", "labelAndArchive"] = Field(
        description="Template name"
    ),
    parameters: Dict[str, Any] = Field(description="Template parameters")
) -> Dict[str, Any]:
    """Create filter from template"""
    
    filter_id = storage.generate_filter_id()
    
    # Build criteria and action based on template
    if template == "fromSender":
        criteria = {"from": parameters.get("senderEmail")}
        action = {
            "addLabelIds": parameters.get("labelIds", []),
            "removeLabelIds": ["INBOX"] if parameters.get("archive") else []
        }
    elif template == "hasAttachment":
        criteria = {"hasAttachment": True}
        action = {"addLabelIds": parameters.get("labelIds", [])}
    else:
        criteria = parameters.get("criteria", {})
        action = parameters.get("action", {})
    
    # Background API sync
    asyncio.create_task(
        email_api_client.api_request(
            endpoint="filters/template",
            method="POST",
            data={
                "template": template,
                "parameters": parameters
            }
        )
    )
    
    logger.info(f"✅ Filter created from template: {filter_id}")
    return {
        "status": "success",
        "filter_id": filter_id,
        "template": template,
        "criteria": criteria,
        "action": action
    }


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ENTERPRISE EMAIL MCP SERVER")
    print("=" * 80)
    print("📡 Mode: Hybrid (Local Storage + Background API Sync)")
    print("🏠 Storage: Local JSON")
    print("☁️  Backend: http://localhost:12005")
    print("=" * 80)
    
    try:
        asyncio.run(
            mcp.run_async(
                transport="streamable-http",
                host="0.0.0.0",
                port=12005,
            )
        )
    except KeyboardInterrupt:
        print("\\n📴 Shutting down Email MCP Server...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise