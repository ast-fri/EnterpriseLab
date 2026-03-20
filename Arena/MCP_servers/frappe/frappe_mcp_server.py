"""
Frappe HRMS MCP Server - Streamable HTTP Version
Provides tools to interact with Frappe HRMS via MCP protocol over HTTP
"""

import asyncio
import json
import logging
from datetime import datetime, date
from typing import Any
from fastmcp import FastMCP
import httpx
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frappe-hrms-mcp")

# Frappe HRMS instance configuration from environment variables
FRAPPE_URL = os.getenv("FRAPPE_URL", "http://localhost:8084")
API_KEY = os.getenv("FRAPPE_API_KEY", "71f6e1bb8d5851b")
API_SECRET = os.getenv("FRAPPE_API_SECRET", "4754019131a21ec")

# You can also use username/password authentication
USERNAME = "Administrator"
PASSWORD = ""  # Add your password here

# Global HTTP client
http_client: httpx.AsyncClient | None = None


def get_auth_headers() -> dict:
    """Get authentication headers for Frappe API requests."""
    if API_KEY and API_SECRET:
        return {
            "Authorization": f"token {API_KEY}:{API_SECRET}",
            "Content-Type": "application/json"
        }
    return {"Content-Type": "application/json"}


async def frappe_request(
    method: str,
    endpoint: str,
    data: dict | None = None,
    params: dict | None = None
) -> dict:
    """Make an authenticated request to Frappe API."""
    global http_client
    
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=30.0)
    
    url = f"{FRAPPE_URL}{endpoint}"
    headers = get_auth_headers()
    
    # If using username/password, login first
    if not (API_KEY and API_SECRET) and USERNAME and PASSWORD:
        try:
            login_response = await http_client.post(
                f"{FRAPPE_URL}/api/method/login",
                data={"usr": USERNAME, "pwd": PASSWORD}
            )
            login_response.raise_for_status()
        except Exception as e:
            return {"error": f"Login failed: {str(e)}"}
    
    try:
        if method.upper() == "GET":
            response = await http_client.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            response = await http_client.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = await http_client.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = await http_client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "status_code": e.response.status_code
        }
    except Exception as e:
        return {"error": str(e)}


async def get_current_employee() -> str | None:
    """Get the current logged-in user's employee ID."""
    result = await frappe_request(
        "GET",
        "/api/method/frappe.auth.get_logged_user"
    )
    
    if "message" in result:
        user = result["message"]
        # Get employee linked to this user
        emp_result = await frappe_request(
            "GET",
            "/api/resource/Employee",
            params={"filters": json.dumps([["user_id", "=", user]]), "limit_page_length": 1}
        )
        
        if emp_result.get("data") and len(emp_result["data"]) > 0:
            return emp_result["data"][0].get("name")
    
    return None


# Initialize FastMCP server
mcp = FastMCP("Frappe HRMS MCP Server")


# Resources
@mcp.resource("hrms://my-profile")
async def get_my_profile_resource() -> str:
    """Current user's employee profile and information"""
    try:
        emp_id = await get_current_employee()
        if emp_id:
            result = await frappe_request("GET", f"/api/resource/Employee/{emp_id}")
            return json.dumps(result, indent=2)
        return json.dumps({"error": "Could not find employee profile for current user"}, indent=2)
    except Exception as e:
        logger.error(f"Error in get_my_profile_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("hrms://my-leaves")
async def get_my_leaves_resource() -> str:
    """Current user's leave balances"""
    try:
        emp_id = await get_current_employee()
        if emp_id:
            result = await frappe_request(
                "GET",
                "/api/resource/Leave Allocation",
                params={
                    "filters": json.dumps([["employee", "=", emp_id]]),
                    "fields": '["leave_type","total_leaves_allocated","unused_leaves","new_leaves_allocated","from_date","to_date"]'
                }
            )
            return json.dumps(result, indent=2)
        return json.dumps({"error": "Could not find employee profile for current user"}, indent=2)
    except Exception as e:
        logger.error(f"Error in get_my_leaves_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("hrms://employees")
async def get_employees_resource() -> str:
    """List of all employees with basic information"""
    try:
        result = await frappe_request(
            "GET", 
            "/api/resource/Employee",
            params={"fields": '["name","employee_name","department","designation","status"]', "limit_page_length": 100}
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_employees_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("hrms://leave-summary")
async def get_leave_summary_resource() -> str:
    """Overview of leave balances and applications"""
    try:
        result = await frappe_request(
            "GET",
            "/api/resource/Leave Application",
            params={"fields": '["name","employee","employee_name","leave_type","from_date","to_date","status"]', "limit_page_length": 50}
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_leave_summary_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("hrms://attendance-today")
async def get_attendance_today_resource() -> str:
    """Attendance records for today"""
    try:
        today = date.today().isoformat()
        result = await frappe_request(
            "GET",
            "/api/resource/Attendance",
            params={"filters": json.dumps([["attendance_date", "=", today]]), "limit_page_length": 100}
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_attendance_today_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("hrms://payroll-summary")
async def get_payroll_summary_resource() -> str:
    """Current payroll information"""
    try:
        result = await frappe_request(
            "GET",
            "/api/resource/Payroll Entry",
            params={"fields": '["name","payroll_frequency","start_date","end_date","status"]', "limit_page_length": 20}
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_payroll_summary_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("hrms://job-openings")
async def get_job_openings_resource() -> str:
    """Current job openings"""
    try:
        result = await frappe_request(
            "GET",
            "/api/resource/Job Opening",
            params={
                "filters": json.dumps([["status", "=", "Open"]]),
                "fields": '["name","job_title","designation","department","status","posted_on"]',
                "limit_page_length": 50
            }
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_job_openings_resource: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


# Tools - Current User
@mcp.tool()
async def get_my_profile() -> str:
    """Get the current logged-in user's employee profile"""
    try:
        emp_id = await get_current_employee()
        if not emp_id:
            return json.dumps({"error": "Could not find employee profile for current user"}, indent=2)
        
        result = await frappe_request("GET", f"/api/resource/Employee/{emp_id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_my_profile: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_my_leave_balance(leave_type: str = None) -> str:
    """Get leave balance for the current logged-in user. Use this when user asks about 'my' leaves.
    
    Args:
        leave_type: Optional specific leave type (e.g., 'Casual Leave', 'Sick Leave', 'Privilege Leave')
    """
    try:
        emp_id = await get_current_employee()
        if not emp_id:
            return json.dumps({"error": "Could not find employee profile for current user"}, indent=2)
        
        filters = [["employee", "=", emp_id]]
        if leave_type:
            filters.append(["leave_type", "=", leave_type])
        
        result = await frappe_request(
            "GET",
            "/api/resource/Leave Allocation",
            params={
                "filters": json.dumps(filters),
                "fields": '["leave_type","total_leaves_allocated","unused_leaves","new_leaves_allocated","from_date","to_date"]'
            }
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_my_leave_balance: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Employee Management
@mcp.tool()
async def get_employee(employee_id: str) -> str:
    """Get detailed information about a specific employee
    
    Args:
        employee_id: Employee ID or name
    """
    try:
        result = await frappe_request("GET", f"/api/resource/Employee/{employee_id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_employee: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def search_employees(department: str = None, designation: str = None, status: str = None, limit: int = 20) -> str:
    """Search employees by department, designation, or status
    
    Args:
        department: Department name
        designation: Designation/role
        status: Employment status (Active, Left, etc.)
        limit: Maximum results (default: 20)
    """
    try:
        filters = []
        if department:
            filters.append(["department", "=", department])
        if designation:
            filters.append(["designation", "=", designation])
        if status:
            filters.append(["status", "=", status])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Employee", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in search_employees: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Leave Management
@mcp.tool()
async def get_leave_balance(employee_id: str, leave_type: str = None) -> str:
    """Get leave balance for a specific employee (use get_my_leave_balance for current user)
    
    Args:
        employee_id: Employee ID
        leave_type: Optional specific leave type
    """
    try:
        filters = [["employee", "=", employee_id]]
        if leave_type:
            filters.append(["leave_type", "=", leave_type])
        
        result = await frappe_request(
            "GET",
            "/api/resource/Leave Allocation",
            params={
                "filters": json.dumps(filters),
                "fields": '["leave_type","total_leaves_allocated","unused_leaves","new_leaves_allocated","from_date","to_date"]'
            }
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_leave_balance: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def apply_leave(leave_type: str, from_date: str, to_date: str, employee_id: str = None, half_day: bool = False, reason: str = "") -> str:
    """Create a new leave application. If no employee_id provided, applies for current user.
    
    Args:
        leave_type: Leave type (e.g., 'Casual Leave', 'Sick Leave', 'Privilege Leave')
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        employee_id: Optional Employee ID (defaults to current user)
        half_day: Is this a half day leave?
        reason: Reason for leave
    """
    try:
        # Get employee ID from current user if not provided
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        data = {
            "doctype": "Leave Application",
            "employee": employee_id,
            "leave_type": leave_type,
            "from_date": from_date,
            "to_date": to_date,
            "half_day": 1 if half_day else 0,
            "description": reason,
        }
        result = await frappe_request("POST", "/api/resource/Leave Application", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in apply_leave: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_leave_applications(employee_id: str = None, status: str = None, from_date: str = None, to_date: str = None, limit: int = 20) -> str:
    """Get leave applications with optional filters
    
    Args:
        employee_id: Filter by employee
        status: Filter by status (Open, Approved, Rejected)
        from_date: From date (YYYY-MM-DD)
        to_date: To date (YYYY-MM-DD)
        limit: Maximum results (default: 20)
    """
    try:
        filters = []
        if employee_id:
            filters.append(["employee", "=", employee_id])
        if status:
            filters.append(["status", "=", status])
        if from_date:
            filters.append(["from_date", ">=", from_date])
        if to_date:
            filters.append(["to_date", "<=", to_date])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Leave Application", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_leave_applications: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def approve_leave(leave_application_id: str) -> str:
    """Approve a leave application
    
    Args:
        leave_application_id: Leave Application ID
    """
    try:
        result = await frappe_request(
            "PUT",
            f"/api/resource/Leave Application/{leave_application_id}",
            data={"status": "Approved"}
        )
        # Submit the document
        submit_result = await frappe_request(
            "POST",
            f"/api/method/frappe.client.submit",
            data={"doc": {"doctype": "Leave Application", "name": leave_application_id}}
        )
        return json.dumps(submit_result, indent=2)
    except Exception as e:
        logger.error(f"Error in approve_leave: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Attendance Management
@mcp.tool()
async def mark_attendance(status: str, employee_id: str = None, attendance_date: str = None) -> str:
    """Mark attendance for an employee
    
    Args:
        status: Status (Present, Absent, Half Day, On Leave, Work From Home)
        employee_id: Employee ID (optional, defaults to current user)
        attendance_date: Date (YYYY-MM-DD), defaults to today
    """
    try:
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        data = {
            "doctype": "Attendance",
            "employee": employee_id,
            "attendance_date": attendance_date or date.today().isoformat(),
            "status": status,
        }
        result = await frappe_request("POST", "/api/resource/Attendance", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in mark_attendance: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_attendance(employee_id: str = None, from_date: str = None, to_date: str = None, status: str = None, limit: int = 50) -> str:
    """Get attendance records with filters
    
    Args:
        employee_id: Filter by employee
        from_date: From date (YYYY-MM-DD)
        to_date: To date (YYYY-MM-DD)
        status: Filter by status
        limit: Maximum results (default: 50)
    """
    try:
        filters = []
        if employee_id:
            filters.append(["employee", "=", employee_id])
        if from_date:
            filters.append(["attendance_date", ">=", from_date])
        if to_date:
            filters.append(["attendance_date", "<=", to_date])
        if status:
            filters.append(["status", "=", status])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Attendance", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_attendance: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def checkin(log_type: str, employee_id: str = None, time: str = None) -> str:
    """Record employee check-in/check-out
    
    Args:
        log_type: IN or OUT
        employee_id: Employee ID (optional, defaults to current user)
        time: Time (YYYY-MM-DD HH:MM:SS), optional - defaults to now
    """
    try:
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        from_time = time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "doctype": "Employee Checkin",
            "employee": employee_id,
            "log_type": log_type,
            "time": from_time,
        }
        result = await frappe_request("POST", "/api/resource/Employee Checkin", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in checkin: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def create_timesheet(time_logs: list, employee_id: str = None) -> str:
    """Create or fill a timesheet entry for an employee
    
    Args:
        time_logs: Array of time log entries with activity_type, from_time, to_time, etc.
        employee_id: Employee ID (optional, defaults to current user)
    """
    try:
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        data = {
            "doctype": "Timesheet",
            "employee": employee_id,
            "time_logs": time_logs
        }
        
        result = await frappe_request("POST", "/api/resource/Timesheet", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in create_timesheet: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_timesheets(employee_id: str = None, from_date: str = None, to_date: str = None, status: str = None, limit: int = 20) -> str:
    """Get timesheet entries with optional filters
    
    Args:
        employee_id: Filter by employee (optional, defaults to current user)
        from_date: From date (YYYY-MM-DD)
        to_date: To date (YYYY-MM-DD)
        status: Filter by status (Draft, Submitted, etc.)
        limit: Maximum results (default: 20)
    """
    try:
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        filters = [["employee", "=", employee_id]]
        
        if from_date:
            filters.append(["start_date", ">=", from_date])
        if to_date:
            filters.append(["end_date", "<=", to_date])
        if status:
            filters.append(["status", "=", status])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Timesheet", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_timesheets: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Payroll
@mcp.tool()
async def get_salary_slip(employee_id: str = None, month: str = None) -> str:
    """Get salary slip for an employee
    
    Args:
        employee_id: Employee ID (optional, defaults to current user)
        month: Month (MM-YYYY)
    """
    try:
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        filters = [["employee", "=", employee_id]]
        if month:
            filters.append(["payroll_period", "like", f"%{month}%"])
        
        params = {
            "filters": json.dumps(filters),
            "limit_page_length": 10
        }
        result = await frappe_request("GET", "/api/resource/Salary Slip", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_salary_slip: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_payroll_entry(payroll_frequency: str = None, status: str = None, limit: int = 10) -> str:
    """Get payroll entries with filters
    
    Args:
        payroll_frequency: Monthly, Fortnightly, Weekly
        status: Draft, Submitted, Cancelled
        limit: Maximum results (default: 10)
    """
    try:
        filters = []
        if payroll_frequency:
            filters.append(["payroll_frequency", "=", payroll_frequency])
        if status:
            filters.append(["docstatus", "=", status])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Payroll Entry", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_payroll_entry: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Expense Claims
@mcp.tool()
async def create_expense_claim(expense_type: str, amount: float, employee_id: str = None, expense_date: str = None, description: str = "") -> str:
    """Create a new expense claim
    
    Args:
        expense_type: Type of expense
        amount: Claim amount
        employee_id: Employee ID (optional, defaults to current user)
        expense_date: Date of expense (YYYY-MM-DD)
        description: Description of expense
    """
    try:
        if not employee_id:
            employee_id = await get_current_employee()
            if not employee_id:
                return json.dumps({"error": "Could not determine employee ID"}, indent=2)
        
        data = {
            "doctype": "Expense Claim",
            "employee": employee_id,
            "expense_date": expense_date or date.today().isoformat(),
            "expenses": [{
                "expense_type": expense_type,
                "amount": amount,
                "description": description
            }]
        }
        result = await frappe_request("POST", "/api/resource/Expense Claim", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in create_expense_claim: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_expense_claims(employee_id: str = None, status: str = None, limit: int = 20) -> str:
    """Get expense claims with filters
    
    Args:
        employee_id: Filter by employee
        status: Status (Draft, Submitted, Paid, Rejected)
        limit: Maximum results (default: 20)
    """
    try:
        filters = []
        if employee_id:
            filters.append(["employee", "=", employee_id])
        if status:
            filters.append(["approval_status", "=", status])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Expense Claim", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_expense_claims: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Recruitment
@mcp.tool()
async def get_job_openings(status: str = None, designation: str = None, department: str = None, limit: int = 20) -> str:
    """Get current job openings
    
    Args:
        status: Open or Closed
        designation: Filter by designation
        department: Filter by department
        limit: Maximum results (default: 20)
    """
    try:
        filters = []
        if status:
            filters.append(["status", "=", status])
        if designation:
            filters.append(["designation", "=", designation])
        if department:
            filters.append(["department", "=", department])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Job Opening", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_job_openings: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def create_job_opening(job_title: str, designation: str, department: str = "", description: str = "", no_of_positions: int = None, status: str = "Open") -> str:
    """Create a new job opening
    
    Args:
        job_title: Job title
        designation: Designation
        department: Department
        description: Job description
        no_of_positions: Number of positions
        status: Status (Open/Closed), defaults to Open
    """
    try:
        data = {
            "doctype": "Job Opening",
            "job_title": job_title,
            "designation": designation,
            "department": department,
            "description": description,
            "status": status,
        }
        
        if no_of_positions:
            data["no_of_positions"] = no_of_positions
        
        result = await frappe_request("POST", "/api/resource/Job Opening", data=data)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in create_job_opening: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_job_opening_details(job_opening_id: str) -> str:
    """Get detailed information about a specific job opening
    
    Args:
        job_opening_id: Job Opening ID
    """
    try:
        result = await frappe_request("GET", f"/api/resource/Job Opening/{job_opening_id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_job_opening_details: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_job_applicants(job_opening: str = None, status: str = None, limit: int = 20) -> str:
    """Get job applicants with filters
    
    Args:
        job_opening: Filter by job opening
        status: Status (Open, Accepted, Rejected, etc.)
        limit: Maximum results (default: 20)
    """
    try:
        filters = []
        if job_opening:
            filters.append(["job_title", "=", job_opening])
        if status:
            filters.append(["status", "=", status])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Job Applicant", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_job_applicants: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Shifts
@mcp.tool()
async def get_shift_assignments(employee_id: str = None, from_date: str = None, to_date: str = None, limit: int = 20) -> str:
    """Get shift assignments for employees
    
    Args:
        employee_id: Filter by employee
        from_date: From date (YYYY-MM-DD)
        to_date: To date (YYYY-MM-DD)
        limit: Maximum results (default: 20)
    """
    try:
        filters = []
        if employee_id:
            filters.append(["employee", "=", employee_id])
        if from_date:
            filters.append(["start_date", ">=", from_date])
        if to_date:
            filters.append(["end_date", "<=", to_date])
        
        params = {"limit_page_length": limit}
        if filters:
            params["filters"] = json.dumps(filters)
        
        result = await frappe_request("GET", "/api/resource/Shift Assignment", params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_shift_assignments: {str(e)}")
        return f"Error: {str(e)}"


# Tools - Reports
@mcp.tool()
async def get_hrms_report(report_name: str, filters: dict = None) -> str:
    """Generate HR reports (attendance, leave, payroll, etc.)
    
    Args:
        report_name: Report name (e.g., 'Monthly Attendance Sheet', 'Employee Leave Balance')
        filters: Report-specific filters
    """
    try:
        result = await frappe_request(
            "POST",
            f"/api/method/frappe.desk.query_report.run",
            data={"report_name": report_name, "filters": filters or {}}
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_hrms_report: {str(e)}")
        return f"Error: {str(e)}"

        
if __name__ == "__main__":
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=12013,
        )
    )