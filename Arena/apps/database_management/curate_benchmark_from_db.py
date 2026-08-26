from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


BASE = Path("/home/fripl/vharsh/EnterpriseLab/Arena/apps/database_management")
SRC_DIR = BASE / "fetched_data"
DST_DIR = BASE / "fetched_data_from_db"
TEMPLATE_DIR = BASE / "fetched_data_from_db_backup"
SCHEMA_DIR = BASE / "fetched_schemas"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_schema_map(path: Path) -> dict[str, list[dict]]:
    payload = load_json(path)
    tables = payload.get("tables", [])
    return {table["name"]: table.get("columns", []) for table in tables}


def default_from_schema(column: dict) -> object:
    default = column.get("column_default")
    if default is not None:
        return str(default)

    nullable = column.get("is_nullable") == "YES"
    dtype = column.get("data_type")

    if dtype in {"json", "object"}:
        return {} if not nullable else None
    if dtype == "array":
        return []
    if dtype in {"tinyint", "smallint", "mediumint", "int", "bigint", "decimal", "double", "float"}:
        return "NULL" if nullable else "0"
    if dtype in {"boolean", "bool"}:
        return False if not nullable else None
    if dtype in {"date", "datetime", "timestamp", "time"}:
        return "NULL"
    return "NULL" if nullable else ""


def row_from_template_or_schema(
    templates: list[dict],
    schema_map: dict[str, list[dict]],
    table_name: str,
    index: int,
    overrides: dict,
) -> dict:
    if templates:
        base = copy.deepcopy(templates[min(index, len(templates) - 1)])
        return merge_on_template(base, overrides)

    row = {}
    for column in schema_map.get(table_name, []):
        row[column["column_name"]] = default_from_schema(column)
    row.update(overrides)
    return row


def stable_uuid(prefix: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{prefix}:{value}"))


def first_last(name: str) -> tuple[str, str]:
    parts = name.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def source_payload(name: str) -> dict:
    return load_json(SRC_DIR / f"{name}.json")


def target_payload(name: str) -> dict:
    file_name = f"{name}.json" if name != "rocket_chat" else "rocket_chat.json"
    return load_json(TEMPLATE_DIR / file_name)


def save_target(name: str, payload: dict) -> None:
    file_name = f"{name}.json" if name != "rocket_chat" else "rocket_chat.json"
    dump_json(DST_DIR / file_name, payload)


def merge_on_template(template, overrides):
    if isinstance(template, dict):
        out = copy.deepcopy(template)
        if not isinstance(overrides, dict):
            return out
        for key in template:
            if key in overrides:
                out[key] = merge_on_template(template[key], overrides[key])
        return out
    if isinstance(template, list):
        return copy.deepcopy(template if overrides is None else overrides)
    return copy.deepcopy(overrides)


def curate_frappe() -> None:
    src = source_payload("frappe")["data"]
    dst = target_payload("frappe")
    schema = build_schema_map(SCHEMA_DIR / "frappe_schema.json")
    tables = dst["tables"]

    employee_templates = tables.get("tabEmployee", [])
    user_templates = tables.get("tabUser", [])
    department_templates = tables.get("tabDepartment", [])
    designation_templates = tables.get("tabDesignation", [])
    leave_templates = tables.get("tabLeave Application", [])
    attendance_templates = tables.get("tabAttendance", [])
    holiday_list_templates = tables.get("tabHoliday List", [])
    holiday_templates = tables.get("tabHoliday", [])
    shift_templates = tables.get("tabShift Type", [])
    leave_type_templates = tables.get("tabLeave Type", [])
    company_templates = tables.get("tabCompany", [])

    explicit_users = {user["email"]: user for user in src["users"]}
    tables["tabUser"] = []
    for i, employee in enumerate(src["employees"]):
        email = employee["user_id"]
        full_name = employee["employee_name"]
        source_user = explicit_users.get(email, {})
        creation = source_user.get("creation", f"{employee['date_of_joining']} 09:00:00.000000")
        tables["tabUser"].append(
            row_from_template_or_schema(
                user_templates,
                schema,
                "tabUser",
                i,
                {
                    "name": email,
                    "email": email,
                    "full_name": full_name,
                    "enabled": "1",
                    "user_type": "System User",
                    "creation": creation,
                    "username": email.split("@")[0],
                    "first_name": employee["first_name"],
                    "last_name": employee["last_name"],
                    "mobile_no": "NULL",
                    "send_welcome_email": "1",
                    "is_active": "1" if "is_active" in (user_templates[0] if user_templates else {}) else None,
                },
            )
        )

    tables["tabEmployee"] = [
        row_from_template_or_schema(
            employee_templates,
            schema,
            "tabEmployee",
            i,
            {
                "name": employee["name"],
                "employee": employee["name"],
                "employee_name": employee["employee_name"],
                "first_name": employee["first_name"],
                "last_name": employee["last_name"],
                "designation": employee["designation"],
                "department": employee["department"],
                "company": employee["company"],
                "status": employee["status"],
                "date_of_joining": employee["date_of_joining"],
                "user_id": employee["user_id"],
                "company_email": employee["user_id"],
                "personal_email": employee["user_id"],
                "prefered_contact_email": employee["user_id"],
                "prefered_email": employee["user_id"],
                "gender": "NULL",
                "create_user_permission": "1",
            },
        )
        for i, employee in enumerate(src["employees"])
    ]

    tables["tabDepartment"] = [
        row_from_template_or_schema(
            department_templates,
            schema,
            "tabDepartment",
            i,
            {
                "name": department.get("name", department["department_name"]),
                "department_name": department["department_name"],
                "company": department.get("company", "Enterprise Arena (Demo)"),
                "is_group": str(department.get("is_group", 0)),
                "parent_department": department.get("parent_department", "All Departments"),
                "disabled": str(department.get("disabled", 0)),
            },
        )
        for i, department in enumerate(src["departments"])
    ]

    tables["tabDesignation"] = [
        row_from_template_or_schema(
            designation_templates,
            schema,
            "tabDesignation",
            i,
            {
                "name": designation.get("name", designation["designation_name"]),
                "designation_name": designation["designation_name"],
                "description": designation.get("description", "NULL"),
            },
        )
        for i, designation in enumerate(src["designations"])
    ]

    if leave_templates or schema.get("tabLeave Application"):
        tables["tabLeave Application"] = [
            row_from_template_or_schema(
                leave_templates,
                schema,
                "tabLeave Application",
                i,
                {
                    "name": leave["name"],
                    "employee": leave["employee"],
                    "employee_name": leave["employee_name"],
                    "leave_type": leave["leave_type"],
                    "from_date": leave["from_date"],
                    "to_date": leave["to_date"],
                    "posting_date": leave["from_date"],
                    "total_leave_days": str(leave["total_leave_days"]),
                    "status": leave["status"],
                    "company": "Enterprise Arena (Demo)",
                    "workflow_state": leave["status"],
                    "docstatus": "1" if leave["status"] == "Approved" else "0",
                },
            )
            for i, leave in enumerate(src["leave_applications"])
        ]

    tables["tabAttendance"] = [
        row_from_template_or_schema(
            attendance_templates,
            schema,
            "tabAttendance",
            i,
            {
                "name": attendance["name"],
                "employee": attendance["employee"],
                "employee_name": attendance["employee_name"],
                "attendance_date": attendance["attendance_date"],
                "status": attendance["status"],
                "company": attendance["company"],
                "docstatus": "1",
            },
        )
        for i, attendance in enumerate(src["attendance"])
    ]

    if holiday_list_templates or schema.get("tabHoliday List"):
        tables["tabHoliday List"] = [
            row_from_template_or_schema(
                holiday_list_templates,
                schema,
                "tabHoliday List",
                i,
                {
                    "name": holiday["holiday_list_name"],
                    "holiday_list_name": holiday["holiday_list_name"],
                    "from_date": holiday["from_date"],
                    "to_date": holiday["to_date"],
                    "total_holidays": "1",
                    "weekly_off": "Sunday",
                },
            )
            for i, holiday in enumerate(src["holiday_lists"])
        ]

    if holiday_templates or schema.get("tabHoliday"):
        tables["tabHoliday"] = [
            row_from_template_or_schema(
                holiday_templates,
                schema,
                "tabHoliday",
                i,
                {
                    "name": stable_uuid("frappe-holiday", holiday["name"]),
                    "holiday_date": holiday["from_date"],
                    "description": holiday["name"],
                    "weekly_off": "0",
                    "parent": holiday["holiday_list_name"],
                    "parentfield": "holidays",
                    "parenttype": "Holiday List",
                },
            )
            for i, holiday in enumerate(src["holiday_lists"])
        ]

    if shift_templates or schema.get("tabShift Type"):
        tables["tabShift Type"] = [
            row_from_template_or_schema(
                shift_templates,
                schema,
                "tabShift Type",
                i,
                {
                    "name": shift["name"],
                    "start_time": shift["start_time"],
                    "end_time": shift["end_time"],
                    "enable_auto_attendance": "0",
                    "determine_check_in_and_check_out": "Strictly based on Log Type in Employee Checkin",
                },
            )
            for i, shift in enumerate(src["shift_types"])
        ]

    if "leave_types" in src and (leave_type_templates or schema.get("tabLeave Type")):
        tables["tabLeave Type"] = [
            row_from_template_or_schema(
                leave_type_templates,
                schema,
                "tabLeave Type",
                i,
                {
                    "name": leave_type.get("name", leave_type["leave_type_name"]),
                    "leave_type_name": leave_type["leave_type_name"],
                    "max_leaves_allowed": str(leave_type.get("max_leaves_allowed", 0)),
                    "is_lwp": str(leave_type.get("is_lwp", 0)),
                },
            )
            for i, leave_type in enumerate(src["leave_types"])
        ]

    if "companies" in src and company_templates:
        tables["tabCompany"] = [
            row_from_template_or_schema(
                company_templates,
                schema,
                "tabCompany",
                i,
                {
                    "name": company["name"],
                    "company_name": company.get("company_name", company["name"]),
                    "abbr": company.get("abbr", "EAD"),
                    "default_currency": company.get("default_currency", "INR"),
                    "country": company.get("country", "India"),
                },
            )
            for i, company in enumerate(src["companies"])
        ]

    save_target("frappe", dst)


def curate_plane() -> None:
    src = source_payload("plane")["data"]
    frappe_src = source_payload("frappe")["data"]
    dst = target_payload("plane")
    tables = dst["tables"]

    user_templates = tables["users"]
    workspace_templates = tables["workspaces"]
    project_templates = tables["projects"]
    state_templates = tables["states"]
    issue_templates = tables["issues"]
    cycle_templates = tables["cycles"]
    module_templates = tables["modules"]
    workspace_member_templates = tables["workspace_members"]
    project_member_templates = tables["project_members"]

    roster = [
        "abigail.mitchell@inazuma.com",
        "aarav.mittal@inazuma.com",
        "surya.reddy@inazuma.com",
        "raj.patel@inazuma.com",
        "rahul.khanna@inazuma.com",
        "karan.sharma@inazuma.com",
        "priya.arora@inazuma.com",
        "sameer.malhotra@inazuma.com",
        "ethan.reynolds@inazuma.com",
    ]
    frappe_users = {user["email"]: user for user in frappe_src["users"]}
    frappe_employees_by_email = {employee["user_id"]: employee for employee in frappe_src["employees"]}
    plane_user_ids = [user["id"] for user in user_templates]

    email_to_plane_id = dict(zip(roster, plane_user_ids))
    aarav_id = email_to_plane_id["aarav.mittal@inazuma.com"]
    surya_id = email_to_plane_id["surya.reddy@inazuma.com"]
    project_leads = {
        "Billing Gateway": email_to_plane_id["surya.reddy@inazuma.com"],
        "Document Collaboration Hub": email_to_plane_id["raj.patel@inazuma.com"],
        "Revenue Intelligence Dashboard": email_to_plane_id["sameer.malhotra@inazuma.com"],
        "Customer Onboarding Workspace": email_to_plane_id["sameer.malhotra@inazuma.com"],
        "Field Service Control Tower": email_to_plane_id["ethan.reynolds@inazuma.com"],
        "Support Operations Portal": email_to_plane_id["abigail.mitchell@inazuma.com"],
    }

    tables["users"] = []
    for i, email in enumerate(roster):
        user = frappe_users.get(email)
        if user:
            full_name = user["full_name"]
        else:
            full_name = frappe_employees_by_email[email]["employee_name"]
        first_name, last_name = first_last(full_name)
        tables["users"].append(
            row_from_template_or_schema(
                user_templates,
                {},
                "users",
                i,
                {
                    "id": plane_user_ids[i],
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "display_name": email.split("@")[0],
                    "username": email.split("@")[0].replace(".", ""),
                    "is_active": True,
                    "is_staff": email in {"aarav.mittal@inazuma.com", "abigail.mitchell@inazuma.com"},
                    "is_email_verified": True,
                },
            )
        )

    workspace = src["workspaces"][0]
    tables["workspaces"] = [
        row_from_template_or_schema(
            workspace_templates,
            {},
            "workspaces",
            0,
            {
                "id": workspace["id"],
                "name": workspace["name"],
                "slug": workspace["slug"],
                "created_at": workspace["created_at"],
                "updated_at": workspace["created_at"],
                "created_by_id": aarav_id,
                "owner_id": aarav_id,
                "updated_by_id": aarav_id,
                "timezone": "UTC",
            },
        )
    ]

    project_identifier = {
        "Billing Gateway": "BILL",
        "Document Collaboration Hub": "DOCH",
        "Revenue Intelligence Dashboard": "REVD",
        "Customer Onboarding Workspace": "ONBD",
        "Field Service Control Tower": "FSCT",
        "Support Operations Portal": "SUPO",
    }
    project_by_id = {project["id"]: project for project in src["projects"]}

    tables["projects"] = []
    for i, project in enumerate(src["projects"]):
        tables["projects"].append(
            row_from_template_or_schema(
                project_templates,
                {},
                "projects",
                i,
                {
                    "id": project["id"],
                    "name": project["name"],
                    "description": project["description"],
                    "description_text": project["description"],
                    "identifier": project_identifier[project["name"]],
                    "created_at": project["created_at"],
                    "updated_at": project["created_at"],
                    "created_by_id": aarav_id,
                    "updated_by_id": surya_id,
                    "workspace_id": workspace["id"],
                    "project_lead_id": project_leads[project["name"]],
                    "default_state_id": next(
                        issue["state_id"] for issue in src["issues"] if issue["project_id"] == project["id"]
                    ),
                    "slug": None,
                    "cover_image": None,
                    "archive_in": 0,
                    "close_in": 0,
                    "deleted_at": None,
                },
            )
        )

    state_ids = {}
    tables["states"] = []
    for i, project in enumerate(src["projects"]):
        state_id = next(issue["state_id"] for issue in src["issues"] if issue["project_id"] == project["id"])
        state_ids[project["id"]] = state_id
        tables["states"].append(
            row_from_template_or_schema(
                state_templates,
                {},
                "states",
                i,
                {
                    "id": state_id,
                    "name": "Backlog",
                    "description": f"Primary backlog state for {project['name']}",
                    "color": "#60646C",
                    "slug": "backlog",
                    "created_by_id": aarav_id,
                    "updated_by_id": surya_id,
                    "project_id": project["id"],
                    "workspace_id": workspace["id"],
                    "sequence": 15000,
                    "group": "backlog",
                    "default": True,
                    "is_triage": False,
                    "deleted_at": None,
                },
            )
        )

    tables["issues"] = []
    for i, issue in enumerate(src["issues"]):
        project = project_by_id[issue["project_id"]]
        tables["issues"].append(
            row_from_template_or_schema(
                issue_templates,
                {},
                "issues",
                i,
                {
                    "id": issue["id"],
                    "name": issue["name"],
                    "description": {"blocks": []},
                    "description_html": f"<p>{issue['name']}</p>",
                    "description_stripped": issue["name"],
                    "priority": "none",
                    "sequence_id": i + 1,
                    "created_by_id": project_leads[project["name"]],
                    "project_id": project["id"],
                    "state_id": state_ids[project["id"]],
                    "updated_by_id": project_leads[project["name"]],
                    "workspace_id": workspace["id"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["created_at"],
                    "sort_order": 65535,
                    "is_draft": False,
                    "deleted_at": None,
                },
            )
        )

    tables["cycles"] = []
    for i, cycle in enumerate(src["cycles"]):
        project = project_by_id[cycle["project_id"]]
        tables["cycles"].append(
            row_from_template_or_schema(
                cycle_templates,
                {},
                "cycles",
                i,
                {
                    "id": cycle["id"],
                    "name": cycle["name"],
                    "description": "",
                    "start_date": cycle["start_date"],
                    "end_date": cycle["end_date"],
                    "created_by_id": project_leads[project["name"]],
                    "owned_by_id": project_leads[project["name"]],
                    "project_id": cycle["project_id"],
                    "updated_by_id": project_leads[project["name"]],
                    "workspace_id": workspace["id"],
                    "created_at": cycle["created_at"],
                    "updated_at": cycle["created_at"],
                    "view_props": {},
                    "progress_snapshot": {},
                    "timezone": "UTC",
                    "version": 1,
                },
            )
        )

    tables["modules"] = []
    for i, module in enumerate(src["modules"]):
        project = project_by_id[module["project_id"]]
        tables["modules"].append(
            row_from_template_or_schema(
                module_templates,
                {},
                "modules",
                i,
                {
                    "id": module["id"],
                    "name": module["name"],
                    "description": module["description"],
                    "description_text": module["description"],
                    "description_html": f"<p>{module['description']}</p>",
                    "status": "backlog",
                    "created_by_id": project_leads[project["name"]],
                    "lead_id": project_leads[project["name"]],
                    "project_id": module["project_id"],
                    "updated_by_id": project_leads[project["name"]],
                    "workspace_id": workspace["id"],
                    "created_at": module["created_at"],
                    "updated_at": module["created_at"],
                    "deleted_at": None,
                },
            )
        )

    tables["workspace_members"] = []
    for i, email in enumerate(roster):
        member_id = email_to_plane_id[email]
        role = 20 if email in {"aarav.mittal@inazuma.com", "surya.reddy@inazuma.com", "abigail.mitchell@inazuma.com"} else 15
        tables["workspace_members"].append(
            row_from_template_or_schema(
                workspace_member_templates,
                {},
                "workspace_members",
                i,
                {
                    "id": stable_uuid("plane-workspace-member", email),
                    "role": role,
                    "created_by_id": aarav_id,
                    "member_id": member_id,
                    "updated_by_id": aarav_id,
                    "workspace_id": workspace["id"],
                    "company_role": "" if role == 20 else None,
                    "is_active": True,
                    "deleted_at": None,
                },
            )
        )

    project_membership = {
        "Billing Gateway": [
            "aarav.mittal@inazuma.com",
            "surya.reddy@inazuma.com",
            "rahul.khanna@inazuma.com",
            "anjali.mathew@inazuma.com",
        ],
        "Document Collaboration Hub": [
            "raj.patel@inazuma.com",
            "karan.sharma@inazuma.com",
            "priya.arora@inazuma.com",
        ],
        "Revenue Intelligence Dashboard": [
            "sameer.malhotra@inazuma.com",
            "anjali.mathew@inazuma.com",
            "aarav.mittal@inazuma.com",
        ],
        "Customer Onboarding Workspace": [
            "sameer.malhotra@inazuma.com",
            "surya.reddy@inazuma.com",
            "priya.arora@inazuma.com",
        ],
        "Field Service Control Tower": [
            "ethan.reynolds@inazuma.com",
            "sameer.malhotra@inazuma.com",
            "surya.reddy@inazuma.com",
        ],
        "Support Operations Portal": [
            "abigail.mitchell@inazuma.com",
            "ethan.reynolds@inazuma.com",
            "rahul.khanna@inazuma.com",
            "karan.sharma@inazuma.com",
        ],
    }
    expanded_email_to_plane_id = email_to_plane_id | {
        "anjali.mathew@inazuma.com": stable_uuid("plane-user-virtual", "anjali.mathew@inazuma.com")
    }
    tables["project_members"] = []
    member_rows = []
    for project in src["projects"]:
        for email in project_membership[project["name"]]:
            if email not in email_to_plane_id:
                continue
            member_rows.append((project, email))
    for i, (project, email) in enumerate(member_rows):
        role = 20 if email == "aarav.mittal@inazuma.com" else 15
        tables["project_members"].append(
            row_from_template_or_schema(
                project_member_templates,
                {},
                "project_members",
                i,
                {
                    "id": stable_uuid("plane-project-member", f"{project['id']}:{email}"),
                    "role": role,
                    "created_by_id": aarav_id,
                    "member_id": email_to_plane_id[email],
                    "project_id": project["id"],
                    "updated_by_id": aarav_id,
                    "workspace_id": workspace["id"],
                    "sort_order": 65535,
                    "is_active": True,
                    "deleted_at": None,
                },
            )
        )

    save_target("plane", dst)


def curate_zammad() -> None:
    src = source_payload("zammad")["data"]
    dst = target_payload("zammad")
    schema = build_schema_map(SCHEMA_DIR / "zammad_schema.json")
    tables = dst["tables"]

    user_templates = tables["users"]
    org_templates = tables["organizations"]
    group_templates = tables["groups"]
    role_templates = tables["roles"]
    ticket_templates = tables["tickets"]
    article_templates = tables["ticket_articles"]
    groups_users_templates = tables["groups_users"]
    roles_users_templates = tables["roles_users"]
    tags_templates = tables["tags"]

    tables["organizations"] = [
        row_from_template_or_schema(
            org_templates,
            schema,
            "organizations",
            i,
            {
                "id": org["id"],
                "name": org["name"],
                "shared": True,
                "active": True,
                "vip": False,
                "note": "",
                "updated_by_id": 3,
                "created_by_id": 3,
                "created_at": "2026-02-26T05:16:30.081",
                "updated_at": "2026-02-26T05:16:30.081",
            },
        )
        for i, org in enumerate(src["organizations"])
    ]

    tables["users"] = [
        row_from_template_or_schema(
            user_templates,
            schema,
            "users",
            i,
            {
                "id": user["id"],
                "organization_id": user.get("organization_id", 1 if user["email"].endswith("@inazuma.com") else None),
                "login": user["login"],
                "firstname": user["firstname"],
                "lastname": user["lastname"],
                "email": user["email"],
                "department": user.get("department", ""),
                "active": bool(user.get("active", True)),
                "verified": True,
                "note": "",
                "updated_by_id": 1,
                "created_by_id": 1,
                "created_at": user.get("created_at", "2026-03-01T09:00:00"),
                "updated_at": user.get("updated_at", user.get("created_at", "2026-03-01T09:00:00")),
            },
        )
        for i, user in enumerate(src["users"])
    ]

    tables["groups"] = [
        row_from_template_or_schema(
            group_templates,
            schema,
            "groups",
            i,
            {
                "id": group["id"],
                "name": group["name"],
                "name_last": group["name"],
                "shared": True if "shared" in group_templates[0] else None,
                "active": True,
                "note": group.get("note", ""),
                "updated_by_id": 1,
                "created_by_id": 1,
            },
        )
        for i, group in enumerate(src["groups"])
    ]

    tables["roles"] = [
        row_from_template_or_schema(
            role_templates,
            schema,
            "roles",
            i,
            {
                "id": role["id"],
                "name": role["name"],
                "active": True,
                "note": role.get("note", ""),
                "updated_by_id": 1,
                "created_by_id": 1,
            },
        )
        for i, role in enumerate(src["roles"])
    ]

    tables["tickets"] = [
        row_from_template_or_schema(
            ticket_templates,
            schema,
            "tickets",
            i,
            {
                "id": ticket["id"],
                "group_id": ticket["group_id"],
                "priority_id": ticket["priority_id"],
                "state_id": ticket["state_id"],
                "organization_id": ticket["organization_id"],
                "number": ticket["number"],
                "title": ticket["title"],
                "owner_id": ticket["owner_id"],
                "customer_id": ticket["customer_id"],
                "note": ticket.get("note"),
                "article_count": ticket["article_count"],
                "create_article_type_id": 1,
                "create_article_sender_id": 2,
                "updated_by_id": ticket["owner_id"],
                "created_by_id": ticket["customer_id"],
                "created_at": ticket["created_at"],
                "updated_at": ticket["updated_at"],
            },
        )
        for i, ticket in enumerate(src["tickets"])
    ]

    tables["ticket_articles"] = [
        row_from_template_or_schema(
            article_templates,
            schema,
            "ticket_articles",
            i,
            {
                "id": article["id"],
                "ticket_id": article["ticket_id"],
                "type_id": 1,
                "sender_id": 2 if article["internal"] else 3,
                "from": "Inazuma Support <support@inazuma.com>",
                "to": "Users::Support",
                "subject": article["subject"],
                "content_type": "text/plain",
                "body": article["body"],
                "internal": bool(article["internal"]),
                "updated_by_id": article["created_by_id"],
                "created_by_id": article["created_by_id"],
                "origin_by_id": article["origin_by_id"],
                "created_at": article["created_at"],
                "updated_at": article["created_at"],
            },
        )
        for i, article in enumerate(src["ticket_articles"])
    ]

    internal_users = [4, 5, 8, 9, 10, 12, 13]
    role_assignments = [
        {"user_id": 5, "role_id": 1},
        {"user_id": 4, "role_id": 2},
        {"user_id": 8, "role_id": 2},
        {"user_id": 9, "role_id": 2},
        {"user_id": 10, "role_id": 2},
    ]
    tables["roles_users"] = [
        row_from_template_or_schema(
            roles_users_templates,
            schema,
            "roles_users",
            i,
            assignment,
        )
        for i, assignment in enumerate(role_assignments)
    ]

    group_assignments = [
        {"user_id": 4, "group_id": 2, "access": "full"},
        {"user_id": 5, "group_id": 2, "access": "full"},
        {"user_id": 10, "group_id": 3, "access": "full"},
    ]
    tables["groups_users"] = [
        row_from_template_or_schema(
            groups_users_templates,
            schema,
            "groups_users",
            i,
            assignment,
        )
        for i, assignment in enumerate(group_assignments)
    ]

    if src.get("tags") and tags_templates:
        tables["tags"] = [
            row_from_template_or_schema(
                tags_templates,
                schema,
                "tags",
                i,
                {
                    "id": i + 1,
                    "created_by_id": 1,
                },
            )
            for i, tag in enumerate(src["tags"])
        ]

    save_target("zammad", dst)


def curate_dolibarr() -> None:
    src = source_payload("dolibarr")["data"]
    dst = target_payload("dolibarr")
    schema = build_schema_map(SCHEMA_DIR / "dolibarr_schema.json")
    tables = dst["tables"]

    societe_templates = tables["llx_societe"]
    facture_templates = tables["llx_facture"]
    propal_templates = tables["llx_propal"]
    projet_templates = tables["llx_projet"]
    product_templates = tables["llx_product"]
    socpeople_templates = tables.get("llx_socpeople", [])
    commande_templates = tables.get("llx_commande", [])

    tables["llx_societe"] = []
    for i, company in enumerate(src["companies"]):
        tables["llx_societe"].append(
            row_from_template_or_schema(
                societe_templates,
                schema,
                "llx_societe",
                i,
                {
                    "rowid": str(company["rowid"]),
                    "nom": company["nom"],
                    "email": company.get("email", "NULL"),
                    "phone": company.get("phone", "NULL"),
                    "client": str(company.get("client", 1)),
                    "fournisseur": str(company.get("fournisseur", 0)),
                    "status": "1",
                    "statut": "1",
                    "code_client": company.get("code_client", f"CU26-{company['rowid']}"),
                },
            )
        )

    if socpeople_templates or schema.get("llx_socpeople"):
        tables["llx_socpeople"] = []
        for i, contact in enumerate(src["contacts"]):
            tables["llx_socpeople"].append(
                row_from_template_or_schema(
                    socpeople_templates,
                    schema,
                    "llx_socpeople",
                    i,
                    {
                        "rowid": str(contact["rowid"]),
                        "fk_soc": str(contact["fk_soc"]),
                        "entity": "1",
                        "datec": "2026-03-01 09:00:00",
                        "tms": "2026-03-01 09:00:00",
                        "lastname": contact["lastname"],
                        "firstname": contact["firstname"],
                        "poste": contact.get("poste", "NULL"),
                        "email": contact.get("email", "NULL"),
                        "phone": contact.get("phone", "NULL"),
                        "statut": "1",
                    },
                )
            )

    tables["llx_product"] = []
    for i, product in enumerate(src["products"]):
        tables["llx_product"].append(
            row_from_template_or_schema(
                product_templates,
                schema,
                "llx_product",
                i,
                {
                    "rowid": str(product["rowid"]),
                    "ref": product["ref"],
                    "label": product["label"],
                    "description": product.get("description", "NULL"),
                    "price": f"{product['price']:.8f}",
                    "price_ttc": f"{product.get('price_ttc', product['price']):.8f}",
                    "entity": "1",
                    "tms": "2026-03-01 09:00:00",
                    "datec": "2026-03-01 09:00:00",
                    "fk_user_author": "1",
                    "fk_user_modif": "1",
                    "stockable_product": "1",
                    "hidden": "0",
                },
            )
        )

    tables["llx_facture"] = []
    for i, invoice in enumerate(src["invoices"]):
        tables["llx_facture"].append(
            row_from_template_or_schema(
                facture_templates,
                schema,
                "llx_facture",
                i,
                {
                    "rowid": str(invoice["rowid"]),
                    "ref": invoice["ref"],
                    "fk_soc": str(invoice["fk_soc"]),
                    "datec": invoice.get("datec", "2026-03-01 09:00:00"),
                    "datef": invoice.get("datef", "2026-03-01"),
                    "tms": invoice.get("datec", "2026-03-01 09:00:00"),
                    "paye": str(invoice.get("paye", 0)),
                    "total_ht": f"{float(invoice['total_ht']):.8f}",
                    "total_ttc": f"{float(invoice['total_ttc']):.8f}",
                    "fk_statut": str(invoice.get("fk_statut", 1)),
                    "fk_user_author": "1",
                    "multicurrency_code": "EUR",
                },
            )
        )

    if commande_templates or schema.get("llx_commande"):
        tables["llx_commande"] = []
        for i, order in enumerate(src["orders"]):
            tables["llx_commande"].append(
                row_from_template_or_schema(
                    commande_templates,
                    schema,
                    "llx_commande",
                    i,
                    {
                        "rowid": str(order["rowid"]),
                        "ref": order["ref"],
                        "fk_soc": str(order["fk_soc"]),
                        "date_creation": order.get("date_creation", "2026-03-01 09:00:00"),
                        "date_commande": order.get("date_commande", "2026-03-01"),
                        "tms": order.get("date_creation", "2026-03-01 09:00:00"),
                        "total_ht": f"{float(order['total_ht']):.8f}",
                        "total_ttc": f"{float(order['total_ttc']):.8f}",
                        "fk_statut": str(order.get("fk_statut", 1)),
                        "fk_user_author": "1",
                    },
                )
            )

    tables["llx_propal"] = []
    for i, proposal in enumerate(src["proposals"]):
        tables["llx_propal"].append(
            row_from_template_or_schema(
                propal_templates,
                schema,
                "llx_propal",
                i,
                {
                    "rowid": str(proposal["rowid"]),
                    "ref": proposal["ref"],
                    "fk_soc": str(proposal["fk_soc"]),
                    "datec": proposal.get("datec", "2026-03-01 09:00:00"),
                    "datep": proposal.get("datep", "2026-03-01"),
                    "tms": proposal.get("datec", "2026-03-01 09:00:00"),
                    "total_ht": f"{float(proposal['total_ht']):.8f}",
                    "total_ttc": f"{float(proposal['total_ttc']):.8f}",
                    "fk_statut": str(proposal.get("fk_statut", 2)),
                    "fk_user_author": "1",
                    "note_private": proposal.get("note_private", ""),
                    "note_public": proposal.get("note_public", ""),
                },
            )
        )

    company_ids = {company["rowid"]: company for company in src["companies"]}
    tables["llx_projet"] = []
    for i, project in enumerate(src["projects"]):
        tables["llx_projet"].append(
            row_from_template_or_schema(
                projet_templates,
                schema,
                "llx_projet",
                i,
                {
                    "rowid": str(project["rowid"]),
                    "ref": project["ref"],
                    "fk_soc": str(project["fk_soc"]),
                    "datec": project.get("datec", "2026-03-01 09:00:00"),
                    "tms": project.get("datec", "2026-03-01 09:00:00"),
                    "dateo": project.get("dateo", "2026-03-01"),
                    "datee": project.get("datee", "2026-06-30"),
                    "title": project["title"],
                    "description": project.get("description", "NULL"),
                    "fk_user_creat": "1",
                    "fk_user_modif": "1",
                    "fk_statut": str(project.get("fk_statut", 1)),
                    "opp_percent": str(project.get("opp_percent", "50.00")),
                    "usage_opportunity": "1",
                    "usage_task": "1",
                    "usage_bill_time": "1",
                },
            )
        )

    save_target("dolibarr", dst)


def curate_rocket_chat() -> None:
    src = source_payload("rocketchat")["data"]
    dst = target_payload("rocket_chat")
    collections = dst["collections"]

    room_templates = collections["rocketchat_room"]
    message_templates = collections["rocketchat_message"]
    subscription_templates = collections["rocketchat_subscription"]
    role_templates = collections["rocketchat_roles"]

    channel_template = next(room for room in room_templates if room.get("t") == "c")
    dm_template = next(room for room in room_templates if room.get("t") == "d")
    message_template = message_templates[0]
    subscription_template = subscription_templates[0]
    role_template = role_templates[0]

    collections["users"] = copy.deepcopy(src["users"])

    last_message_by_room = {}
    for message in src["recent_messages"]:
        last_message_by_room[message["rid"]] = {
            "alias": "",
            "msg": message["msg"],
            "attachments": [],
            "parseUrls": True,
            "groupable": False,
            "ts": message["ts"],
            "u": copy.deepcopy(message["u"]),
            "rid": message["rid"],
            "_id": message["_id"],
            "_updatedAt": message["ts"],
            "urls": [],
            "mentions": [],
            "channels": [],
            "md": [
                {
                    "type": "PARAGRAPH",
                    "value": [
                        {
                            "type": "PLAIN_TEXT",
                            "value": message["msg"],
                        }
                    ],
                }
            ],
        }

    collections["rocketchat_room"] = []
    for room in src["rooms"]:
        template = channel_template if room["t"] == "c" else dm_template
        payload = {
            **copy.deepcopy(room),
            "usernames": [],
            "_updatedAt": room.get("_updatedAt", room["ts"]),
            "lm": last_message_by_room.get(room["_id"], {}).get("ts", room["ts"]),
            "lastMessage": last_message_by_room.get(room["_id"]),
        }
        if room["t"] == "c":
            payload["u"] = {
                "_id": "rocket.cat",
                "username": "rocket.cat",
                "name": "Rocket.Cat",
            }
            payload["default"] = room["_id"] == "GENERAL"
        collections["rocketchat_room"].append(merge_on_template(template, payload))

    collections["rocketchat_message"] = []
    for message in src["recent_messages"]:
        payload = copy.deepcopy(message)
        payload["groupable"] = False
        payload["_updatedAt"] = message["ts"]
        collections["rocketchat_message"].append(merge_on_template(message_template, payload))

    collections["rocketchat_subscription"] = []
    for subscription in src["subscriptions"]:
        payload = copy.deepcopy(subscription)
        payload["_updatedAt"] = payload.get("_updatedAt", "2026-04-21T10:35:24.130Z")
        payload["ls"] = payload.get("ls", payload.get("_updatedAt"))
        payload["userMentions"] = payload.get("userMentions", 0)
        payload["groupMentions"] = payload.get("groupMentions", 0)
        collections["rocketchat_subscription"].append(merge_on_template(subscription_template, payload))

    collections["rocketchat_roles"] = []
    for role in src["roles"]:
        payload = copy.deepcopy(role)
        payload["protected"] = payload.get("protected", role["_id"] in {"admin", "user", "bot"})
        payload["_updatedAt"] = payload.get("_updatedAt", "2025-12-12T16:41:20.074Z")
        collections["rocketchat_roles"].append(merge_on_template(role_template, payload))

    collections["rocketchat_integrations"] = copy.deepcopy(src["integrations"])
    save_target("rocket_chat", dst)


def curate_gitlab() -> None:
    src = source_payload("gitlab")["data"]
    dst = target_payload("gitlab")
    schema = build_schema_map(SCHEMA_DIR / "gitlab_schema.json")
    tables = dst["tables"]

    user_templates = tables["users"]
    namespace_templates = tables["namespaces"]
    project_templates = tables["projects"]
    member_templates = tables["members"]
    label_templates = tables["labels"]
    issue_templates = tables["issues"]
    mr_templates = tables["merge_requests"]
    note_templates = tables["notes"]
    milestone_templates = tables.get("milestones", [])
    has_branches = "branches" in tables
    has_commits = "commits" in tables
    has_pipelines = "pipelines" in tables
    branch_templates = tables.get("branches", [])
    commit_templates = tables.get("commits", [])
    pipeline_templates = tables.get("pipelines", [])

    tables["users"] = []
    for i, user in enumerate(src["users"]):
        is_admin = bool(user.get("is_admin", False))
        tables["users"].append(
            row_from_template_or_schema(
                user_templates,
                {},
                "users",
                i,
                {
                    "id": user["id"],
                    "email": user["email"],
                    "username": user["username"],
                    "name": user["name"],
                    "state": user["state"],
                    "created_at": user["created_at"],
                    "updated_at": user.get("last_sign_in_at", user["created_at"]),
                    "admin": is_admin,
                    "organization_id": 1,
                    "confirmed_at": user["created_at"],
                    "preferred_language": "en",
                    "theme_id": 3,
                    "projects_limit": 100000,
                    "can_create_group": True,
                    "can_create_team": False,
                },
            )
        )

    tables["namespaces"] = []
    for i, namespace in enumerate(src["namespaces"]):
        ns_type = "Group" if namespace["kind"] == "group" else "Project"
        tables["namespaces"].append(
            row_from_template_or_schema(
                namespace_templates,
                {},
                "namespaces",
                i,
                {
                    "id": namespace["id"],
                    "name": namespace["name"],
                    "path": namespace["path"],
                    "type": ns_type,
                    "parent_id": namespace["parent_id"],
                    "owner_id": 21,
                    "created_at": "2025-11-06T07:14:22",
                    "updated_at": "2026-06-17T09:20:00",
                    "description": "",
                    "visibility_level": 20 if namespace["parent_id"] is None else 0,
                    "request_access_enabled": True,
                    "organization_id": 1,
                },
            )
        )

    tables["projects"] = []
    for i, project in enumerate(src["projects"]):
        tables["projects"].append(
            row_from_template_or_schema(
                project_templates,
                {},
                "projects",
                i,
                {
                    "id": project["id"],
                    "name": project["name"],
                    "path": project["path"],
                    "description": project["description"],
                    "created_at": project["created_at"],
                    "updated_at": project["updated_at"],
                    "creator_id": project["creator_id"],
                    "namespace_id": project["namespace_id"],
                    "last_activity_at": project["updated_at"],
                    "visibility_level": project["visibility_level"],
                    "archived": project["archived"],
                    "default_branch": project["default_branch"],
                    "issues_template": None,
                    "merge_requests_template": None,
                    "description_html": f"<p dir=\"auto\">{project['description']}</p>",
                    "only_allow_merge_if_pipeline_succeeds": False,
                    "service_desk_enabled": True,
                    "packages_enabled": True,
                    "remove_source_branch_after_merge": True,
                    "project_namespace_id": project["namespace_id"],
                    "organization_id": 1,
                },
            )
        )

    tables["members"] = []
    for i, member in enumerate(src["members"]):
        tables["members"].append(
            row_from_template_or_schema(
                member_templates,
                {},
                "members",
                i,
                {
                    "id": member["id"],
                    "access_level": member["access_level"],
                    "source_id": member["source_id"],
                    "source_type": member["source_type"],
                    "user_id": member["user_id"],
                    "notification_level": 3,
                    "type": "ProjectMember",
                    "created_at": member["created_at"],
                    "updated_at": member["created_at"],
                    "created_by_id": 21,
                    "ldap": False,
                    "override": False,
                    "state": 0,
                    "invite_email_success": True,
                    "member_namespace_id": member["source_id"],
                },
            )
        )

    tables["labels"] = []
    for i, label in enumerate(src["labels"]):
        tables["labels"].append(
            row_from_template_or_schema(
                label_templates,
                {},
                "labels",
                i,
                {
                    "id": label["id"],
                    "title": label["title"],
                    "color": label["color"],
                    "project_id": label["project_id"],
                    "created_at": "2026-01-06T09:05:00",
                    "updated_at": "2026-06-17T09:20:00",
                    "template": False,
                    "type": "ProjectLabel",
                    "archived": False,
                },
            )
        )

    tables["milestones"] = []
    for i, milestone in enumerate(src["milestones"]):
        tables["milestones"].append(
            row_from_template_or_schema(
                milestone_templates,
                schema,
                "milestones",
                i,
                {
                    "id": milestone["id"],
                    "title": milestone["title"],
                    "iid": milestone["iid"],
                    "project_id": milestone["project_id"],
                    "state": 1 if milestone["state"] == "active" else 2,
                    "start_date": milestone["start_date"],
                    "due_date": milestone["due_date"],
                    "created_at": f"{milestone['start_date']}T00:00:00",
                    "updated_at": f"{milestone['due_date']}T23:59:00",
                    "description": "",
                },
            )
        )

    tables["issues"] = []
    for i, issue in enumerate(src["issues"]):
        assignee_id = issue.get("assignee_ids", [None])[0]
        tables["issues"].append(
            row_from_template_or_schema(
                issue_templates,
                {},
                "issues",
                i,
                {
                    "id": issue["id"],
                    "title": issue["title"],
                    "author_id": issue["author_id"],
                    "project_id": issue["project_id"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["created_at"],
                    "description": issue["description"],
                    "milestone_id": issue["milestone_id"],
                    "iid": issue["iid"],
                    "updated_by_id": assignee_id,
                    "description_html": f"<p dir=\"auto\">{issue['description']}</p>",
                    "state_id": 1 if issue["state"] == "opened" else 2,
                    "namespace_id": next(
                        project["namespace_id"] for project in src["projects"] if project["id"] == issue["project_id"]
                    ),
                    "author_id_convert_to_bigint": issue["author_id"],
                    "id_convert_to_bigint": issue["id"],
                    "project_id_convert_to_bigint": issue["project_id"],
                },
            )
        )

    tables["merge_requests"] = []
    for i, mr in enumerate(src["merge_requests"]):
        assignee_id = mr.get("assignee_ids", [None])[0]
        state_id = 1 if mr["state"] == "opened" else 3
        tables["merge_requests"].append(
            row_from_template_or_schema(
                mr_templates,
                {},
                "merge_requests",
                i,
                {
                    "id": mr["id"],
                    "target_branch": mr["target_branch"],
                    "source_branch": mr["source_branch"],
                    "source_project_id": mr["project_id"],
                    "author_id": mr["author_id"],
                    "assignee_id": assignee_id,
                    "title": mr["title"],
                    "created_at": mr["created_at"],
                    "updated_at": mr["created_at"],
                    "milestone_id": mr["milestone_id"],
                    "merge_status": mr["merge_status"],
                    "target_project_id": mr["project_id"],
                    "iid": mr["iid"],
                    "description": mr["title"],
                    "state_id": state_id,
                    "draft": False,
                    "prepared_at": mr["created_at"],
                },
            )
        )

    if has_branches:
        tables["branches"] = []
        for i, branch in enumerate(src["branches"]):
            tables["branches"].append(
                row_from_template_or_schema(
                    branch_templates,
                    {},
                    "branches",
                    i,
                    copy.deepcopy(branch),
                )
            )

    if has_commits:
        tables["commits"] = []
        for i, commit in enumerate(src["commits"]):
            tables["commits"].append(
                row_from_template_or_schema(
                    commit_templates,
                    {},
                    "commits",
                    i,
                    copy.deepcopy(commit),
                )
            )

    if has_pipelines:
        tables["pipelines"] = []
        for i, pipeline in enumerate(src["pipelines"]):
            tables["pipelines"].append(
                row_from_template_or_schema(
                    pipeline_templates,
                    {},
                    "pipelines",
                    i,
                    copy.deepcopy(pipeline),
                )
            )

    tables["notes"] = []
    for i, note in enumerate(src["notes"]):
        payload = copy.deepcopy(note)
        payload.setdefault("note", payload.pop("body"))
        payload.setdefault("created_at", "2026-02-03T12:00:00")
        payload.setdefault("updated_at", payload["created_at"])
        payload.setdefault("project_id", next(
            issue["project_id"] for issue in src["issues"] if issue["iid"] == payload["noteable_iid"]
        ) if payload["noteable_type"] == "Issue" else next(
            mr["project_id"] for mr in src["merge_requests"] if mr["iid"] == payload["noteable_iid"]
        ))
        payload.setdefault("author_id", 21)
        payload.setdefault("system", False)
        payload.setdefault("internal", False)
        payload.setdefault("discussion_id", stable_uuid("gitlab-note", str(payload["id"])))
        payload.setdefault("note_html", f"<p dir=\"auto\">{payload['note']}</p>")
        payload.setdefault("id", note["id"])
        tables["notes"].append(
            row_from_template_or_schema(
                note_templates,
                {},
                "notes",
                i,
                payload,
            )
        )

    save_target("gitlab", dst)


def main() -> None:
    curate_frappe()
    curate_plane()
    curate_gitlab()
    curate_zammad()
    curate_dolibarr()
    curate_rocket_chat()


if __name__ == "__main__":
    main()
