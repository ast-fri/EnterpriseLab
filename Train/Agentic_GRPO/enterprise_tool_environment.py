"""
EnterpriseBench Tool Environment - Production wrapper for your Tools class.

This integrates your existing tools.py into the GRPO training pipeline.
"""

import sys
import tempfile
import shutil
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add your EnterpriseBench path
sys.path.insert(0, "path/to/EnterpriseBench")  # Update this path as needed

from tools import Tools  # Your existing Tools class

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result from executing a tool."""
    success: bool
    output: str
    error: Optional[str] = None


class EnterpriseBenchToolEnvironment:
    """
    Production wrapper for EnterpriseBench tools.

    Integrates your existing Tools class with the GRPO training pipeline.
    Each instance gets isolated workspace for parallel trajectory generation.
    """

    def __init__(self, workspace_base: str = "path/to/enterprise_workspace"):
        """
        Initialize environment with workspace isolation.

        Args:
            workspace_base: Base directory for EnterpriseBench JSON files
        """
        self.workspace_base = workspace_base
        self.tools_instance = Tools()

        # Create isolated temp workspace for this trajectory
        self.temp_workspace = tempfile.mkdtemp(prefix="enterprise_grpo_")
        logger.info(f"Created isolated workspace: {self.temp_workspace}")

        # Map tool names to methods
        self.tool_methods = self._build_tool_map()

    def _build_tool_map(self) -> Dict[str, callable]:
        """Build mapping of tool names to callable methods."""
        # Get all methods from Tools class that are actual tools
        tool_map = {}

        # GitHub tools
        github_tools = [
            "github_list_my_repositories",
            "github_list_issues_of_repository",
            "github_create_repository",
            "github_create_issue",
            "github_get_issue",
            "github_update_repository",
            "github_delete_repository",
            "github_update_issue",
            "github_delete_issue",
            "github_get_repository_contents"
        ]

        # Email tools
        email_tools = [
            "read_email",
            "create_email",
            "update_email",
            "delete_email",
            "list_my_email_threads",
            "list_thread_ids_between_sender_recipient",
            "list_email_ids_in_thread"
        ]

        # Messaging/Collaboration tools
        collab_tools = [
            "send_message",
            "edit_message",
            "delete_message",
            "list_conversation_ids_between_employees",
            "fetch_conversation_by_id"
        ]

        # CRM - Customer Support
        crm_support_tools = [
            "read_customer_support_chat",
            "create_customer_support_chat",
            "update_customer_support_chat",
            "delete_customer_support_chat",
            "read_my_crm_chats",
            "list_customer_support_chats_by_product",
            "list_customer_support_chats_by_customer"
        ]

        # CRM - Products
        crm_product_tools = [
            "create_product",
            "get_product",
            "update_product",
            "delete_product",
            "list_products_by_category"
        ]

        # CRM - Customers
        crm_customer_tools = [
            "get_customer"
        ]

        # CRM - Product Sentiment/Reviews
        crm_sentiment_tools = [
            "create_product_sentiment",
            "get_product_reviews",
            "get_customer_reviews",
            "get_product_sentiment",
            "update_product_sentiment",
            "delete_product_sentiment"
        ]

        # CRM - Sales
        crm_sales_tools = [
            "create_sales_record",
            "get_sales_record",
            "update_sales_record",
            "delete_sales_record",
            "list_sales_by_product",
            "list_sales_by_customer"
        ]

        all_tools = (
            github_tools + email_tools + collab_tools + 
            crm_support_tools + crm_product_tools + crm_customer_tools +
            crm_sentiment_tools + crm_sales_tools
        )

        for tool_name in all_tools:
            if hasattr(self.tools_instance, tool_name):
                tool_map[tool_name] = getattr(self.tools_instance, tool_name)

        logger.info(f"Loaded {len(tool_map)} EnterpriseBench tools")
        return tool_map

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            ToolExecutionResult with output or error
        """
        if tool_name not in self.tool_methods:
            return ToolExecutionResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available tools: {list(self.tool_methods.keys())}"
            )

        try:
            # Call the tool method
            tool_method = self.tool_methods[tool_name]
            result = tool_method(arguments)

            # Convert result to string
            if isinstance(result, list):
                if len(result) == 0:
                    output = "No results found."
                else:
                    # Format as readable JSON
                    import json
                    output = json.dumps(result, indent=2)
            else:
                output = str(result)

            return ToolExecutionResult(
                success=True,
                output=output,
                error=None
            )

        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
            return ToolExecutionResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}"
            )

    def reset(self):
        """Clean up workspace after trajectory generation."""
        try:
            if hasattr(self, 'temp_workspace'):
                shutil.rmtree(self.temp_workspace, ignore_errors=True)
                logger.debug(f"Cleaned up workspace: {self.temp_workspace}")
        except Exception as e:
            logger.warning(f"Failed to clean workspace: {e}")

    @staticmethod
    def get_tool_schema() -> Dict[str, Dict[str, Any]]:
        """
        Get complete tool schema for prompt generation.

        Returns all EnterpriseBench tools with descriptions and argument schemas.
        """
        return {
            # ============================================================
            # GITHUB TOOLS
            # ============================================================
            "github_list_my_repositories": {
                "description": "Lists all GitHub repositories accessible to an employee",
                "args_schema": {
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True}
                }
            },
            "github_list_issues_of_repository": {
                "description": "Lists all issues for a specified GitHub repository",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True}
                }
            },
            "github_create_repository": {
                "description": "Creates a new GitHub repository",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True},
                    "license": {"type": "string", "description": "License type (e.g., MIT, Apache-2.0)", "required": False},
                    "creation_date": {"type": "string", "description": "Creation date", "required": False}
                }
            },
            "github_create_issue": {
                "description": "Creates a new issue in a GitHub repository",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "id": {"type": "string", "description": "Issue ID", "required": False},
                    "title": {"type": "string", "description": "Issue title", "required": False},
                    "description": {"type": "string", "description": "Issue description", "required": False},
                    "status": {"type": "string", "description": "Issue status (Open/Closed)", "required": False},
                    "created_at": {"type": "string", "description": "Creation date", "required": False}
                }
            },
            "github_get_issue": {
                "description": "Retrieves details for a specific issue by ID",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "id": {"type": "string", "description": "Issue ID", "required": True}
                }
            },
            "github_update_repository": {
                "description": "Updates metadata for a GitHub repository",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "emp_id": {"type": "string", "description": "Employee ID", "required": False},
                    "license": {"type": "string", "description": "License type", "required": False},
                    "creation_date": {"type": "string", "description": "Creation date", "required": False}
                }
            },
            "github_delete_repository": {
                "description": "Deletes a GitHub repository if access rights verified",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True}
                }
            },
            "github_update_issue": {
                "description": "Updates an existing issue's details",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "id": {"type": "string", "description": "Issue ID", "required": True},
                    "title": {"type": "string", "description": "Issue title", "required": False},
                    "description": {"type": "string", "description": "Issue description", "required": False},
                    "status": {"type": "string", "description": "Issue status", "required": False}
                }
            },
            "github_delete_issue": {
                "description": "Deletes an issue from a repository",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "id": {"type": "string", "description": "Issue ID", "required": True}
                }
            },
            "github_get_repository_contents": {
                "description": "Retrieves files and metadata inside a GitHub repository",
                "args_schema": {
                    "repo_name": {"type": "string", "description": "Repository name", "required": True},
                    "path": {"type": "string", "description": "Path within repository", "required": False}
                }
            },

            # ============================================================
            # EMAIL TOOLS
            # ============================================================
            "read_email": {
                "description": "Reads a specific email by email ID",
                "args_schema": {
                    "email_id": {"type": "string", "description": "Unique email ID", "required": True}
                }
            },
            "create_email": {
                "description": "Creates and sends a new email",
                "args_schema": {
                    "email_id": {"type": "string", "description": "Unique email ID", "required": False},
                    "thread_id": {"type": "string", "description": "Thread ID", "required": False},
                    "sender_email": {"type": "string", "description": "Sender email address", "required": True},
                    "sender_name": {"type": "string", "description": "Sender name", "required": False},
                    "recipient_email": {"type": "string", "description": "Recipient email", "required": True},
                    "recipient_name": {"type": "string", "description": "Recipient name", "required": False},
                    "subject": {"type": "string", "description": "Email subject", "required": True},
                    "body": {"type": "string", "description": "Email body", "required": True},
                    "date": {"type": "string", "description": "Date", "required": False},
                    "importance": {"type": "string", "description": "Importance (Normal/High/Low)", "required": False},
                    "category": {"type": "string", "description": "Category (INTERNAL/EXTERNAL)", "required": False}
                }
            },
            "update_email": {
                "description": "Updates an existing email's metadata",
                "args_schema": {
                    "email_id": {"type": "string", "description": "Email ID to update", "required": True},
                    "subject": {"type": "string", "description": "New subject", "required": False},
                    "body": {"type": "string", "description": "New body", "required": False},
                    "importance": {"type": "string", "description": "New importance", "required": False}
                }
            },
            "delete_email": {
                "description": "Deletes an email by ID",
                "args_schema": {
                    "email_id": {"type": "string", "description": "Email ID to delete", "required": True}
                }
            },
            "list_my_email_threads": {
                "description": "Lists all email threads for a user",
                "args_schema": {
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True},
                    "start_date": {"type": "string", "description": "Start date filter", "required": False},
                    "end_date": {"type": "string", "description": "End date filter", "required": False},
                    "importance": {"type": "string", "description": "Filter by importance", "required": False}
                }
            },
            "list_thread_ids_between_sender_recipient": {
                "description": "Lists thread IDs between two email addresses",
                "args_schema": {
                    "sender_email": {"type": "string", "description": "Sender email", "required": True},
                    "recipient_email": {"type": "string", "description": "Recipient email", "required": True}
                }
            },
            "list_email_ids_in_thread": {
                "description": "Lists all email IDs in a specific thread",
                "args_schema": {
                    "thread_id": {"type": "string", "description": "Thread ID", "required": True}
                }
            },

            # ============================================================
            # COLLABORATION/MESSAGING TOOLS
            # ============================================================
            "send_message": {
                "description": "Sends a message between two employees",
                "args_schema": {
                    "conversation_id": {"type": "string", "description": "Conversation ID", "required": False},
                    "sender_emp_id": {"type": "string", "description": "Sender employee ID", "required": True},
                    "recipient_emp_id": {"type": "string", "description": "Recipient employee ID", "required": True},
                    "text": {"type": "string", "description": "Message text", "required": True},
                    "category": {"type": "string", "description": "Message category", "required": False},
                    "date": {"type": "string", "description": "Message date", "required": False}
                }
            },
            "edit_message": {
                "description": "Edits an existing message",
                "args_schema": {
                    "conversation_id": {"type": "string", "description": "Conversation ID", "required": True},
                    "text": {"type": "string", "description": "New message text", "required": False},
                    "category": {"type": "string", "description": "New category", "required": False}
                }
            },
            "delete_message": {
                "description": "Deletes a conversation message",
                "args_schema": {
                    "conversation_id": {"type": "string", "description": "Conversation ID", "required": True},
                    "sender_emp_id": {"type": "string", "description": "Sender employee ID", "required": True},
                    "recipient_emp_id": {"type": "string", "description": "Recipient employee ID", "required": True}
                }
            },
            "list_conversation_ids_between_employees": {
                "description": "Lists conversation IDs between two employees",
                "args_schema": {
                    "sender_emp_id": {"type": "string", "description": "First employee ID", "required": True},
                    "recipient_emp_id": {"type": "string", "description": "Second employee ID", "required": True}
                }
            },
            "fetch_conversation_by_id": {
                "description": "Fetches full conversation data by ID",
                "args_schema": {
                    "conversation_id": {"type": "string", "description": "Conversation ID", "required": True}
                }
            },

            # ============================================================
            # CRM - CUSTOMER SUPPORT TOOLS
            # ============================================================
            "read_customer_support_chat": {
                "description": "Reads customer support chat by chat ID",
                "args_schema": {
                    "chat_id": {"type": "string", "description": "Chat ID", "required": True},
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True}
                }
            },
            "create_customer_support_chat": {
                "description": "Creates a new customer support chat record",
                "args_schema": {
                    "chat_id": {"type": "string", "description": "Chat ID", "required": False},
                    "product_id": {"type": "string", "description": "Product ID", "required": True},
                    "customer_id": {"type": "string", "description": "Customer ID", "required": True},
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True},
                    "text": {"type": "string", "description": "Chat text", "required": True},
                    "interaction_date": {"type": "string", "description": "Interaction date", "required": False}
                }
            },
            "update_customer_support_chat": {
                "description": "Updates an existing support chat",
                "args_schema": {
                    "chat_id": {"type": "string", "description": "Chat ID", "required": True},
                    "text": {"type": "string", "description": "New chat text", "required": False},
                    "product_id": {"type": "string", "description": "New product ID", "required": False}
                }
            },
            "delete_customer_support_chat": {
                "description": "Deletes a customer support chat",
                "args_schema": {
                    "chat_id": {"type": "string", "description": "Chat ID", "required": True}
                }
            },
            "read_my_crm_chats": {
                "description": "Lists chat IDs handled by an employee",
                "args_schema": {
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True},
                    "start_date": {"type": "string", "description": "Start date filter", "required": False},
                    "end_date": {"type": "string", "description": "End date filter", "required": False}
                }
            },
            "list_customer_support_chats_by_product": {
                "description": "Lists support chats filtered by product",
                "args_schema": {
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True},
                    "product_id": {"type": "string", "description": "Product ID", "required": True}
                }
            },
            "list_customer_support_chats_by_customer": {
                "description": "Lists support chats filtered by customer",
                "args_schema": {
                    "emp_id": {"type": "string", "description": "Employee ID", "required": True},
                    "customer_id": {"type": "string", "description": "Customer ID", "required": True}
                }
            },

            # ============================================================
            # CRM - PRODUCT TOOLS
            # ============================================================
            "create_product": {
                "description": "Creates a new product entry",
                "args_schema": {
                    "product_id": {"type": "string", "description": "Product ID", "required": True},
                    "product_name": {"type": "string", "description": "Product name", "required": True},
                    "category": {"type": "string", "description": "Product category", "required": False},
                    "actual_price": {"type": "string", "description": "Actual price", "required": False},
                    "discounted_price": {"type": "string", "description": "Discounted price", "required": False}
                }
            },
            "get_product": {
                "description": "Retrieves product details by ID or name pattern",
                "args_schema": {
                    "product_id": {"type": "string", "description": "Product ID", "required": False},
                    "product_name": {"type": "string", "description": "Product name (supports wildcards)", "required": False}
                }
            },
            "update_product": {
                "description": "Updates an existing product",
                "args_schema": {
                    "product_id": {"type": "string", "description": "Product ID", "required": True},
                    "product_name": {"type": "string", "description": "New product name", "required": False},
                    "category": {"type": "string", "description": "New category", "required": False},
                    "actual_price": {"type": "string", "description": "New actual price", "required": False}
                }
            },
            "delete_product": {
                "description": "Deletes a product entry",
                "args_schema": {
                    "product_id": {"type": "string", "description": "Product ID", "required": True}
                }
            },
            "list_products_by_category": {
                "description": "Lists products filtered by category",
                "args_schema": {
                    "category": {"type": "string", "description": "Product category", "required": True}
                }
            },

            # ============================================================
            # CRM - CUSTOMER TOOLS
            # ============================================================
            "get_customer": {
                "description": "Retrieves customer details by ID or name",
                "args_schema": {
                    "customer_id": {"type": "string", "description": "Customer ID", "required": False},
                    "customer_name": {"type": "string", "description": "Customer name", "required": False}
                }
            },

            # ============================================================
            # CRM - PRODUCT SENTIMENT/REVIEW TOOLS
            # ============================================================
            "create_product_sentiment": {
                "description": "Creates a new product review/sentiment entry",
                "args_schema": {
                    "sentiment_id": {"type": "string", "description": "Sentiment ID", "required": False},
                    "product_id": {"type": "string", "description": "Product ID", "required": True},
                    "customer_id": {"type": "string", "description": "Customer ID", "required": True},
                    "review_content": {"type": "string", "description": "Review content", "required": True},
                    "review_date": {"type": "string", "description": "Review date", "required": False}
                }
            },
            "get_product_reviews": {
                "description": "Lists all review sentiment IDs for a product",
                "args_schema": {
                    "product_id": {"type": "string", "description": "Product ID", "required": True}
                }
            },
            "get_customer_reviews": {
                "description": "Lists all review sentiment IDs by a customer",
                "args_schema": {
                    "customer_id": {"type": "string", "description": "Customer ID", "required": True}
                }
            },
            "get_product_sentiment": {
                "description": "Retrieves product sentiment data by sentiment ID",
                "args_schema": {
                    "sentiment_id": {"type": "string", "description": "Sentiment ID", "required": True}
                }
            },
            "update_product_sentiment": {
                "description": "Updates an existing product sentiment entry",
                "args_schema": {
                    "sentiment_id": {"type": "string", "description": "Sentiment ID", "required": True},
                    "review_content": {"type": "string", "description": "New review content", "required": False}
                }
            },
            "delete_product_sentiment": {
                "description": "Deletes a product sentiment entry",
                "args_schema": {
                    "sentiment_id": {"type": "string", "description": "Sentiment ID", "required": True}
                }
            },
        }


def create_enterprise_tool_environment():
    """Factory function for creating isolated environments."""
    return EnterpriseBenchToolEnvironment()