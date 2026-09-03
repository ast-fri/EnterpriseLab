import json
import os
from datetime import date

import frappe
from frappe.utils.password import update_password


SITE = os.environ.get("FRAPPE_SITE", "hrms.localhost")
JSON_PATH = "/workspace/frappe_from_db.json"
CREDENTIALS_PATH = "/workspace/user-credentials.json"
COMPANY = "Enterprise Arena (Demo)"
COMPANY_ABBR = "EAD"


def value(raw, default=None):
    return default if raw in (None, "", "NULL") else raw


def rows(table):
    return export["tables"].get(table, [])


def exists(doctype, name):
    return bool(name and frappe.db.exists(doctype, name))


def ensure_master(doctype, name, **fields):
    if not exists(doctype, name):
        frappe.get_doc({"doctype": doctype, **fields}).insert(ignore_permissions=True)
    return name


def create_company():
    if not exists("Warehouse Type", "Transit"):
        frappe.get_doc(
            {
                "doctype": "Warehouse Type",
                "name": "Transit",
                "description": "Goods moving between warehouses",
            }
        ).insert(ignore_permissions=True)
    if exists("Company", COMPANY):
        return
    frappe.get_doc(
        {
            "doctype": "Company",
            "company_name": COMPANY,
            "abbr": COMPANY_ABBR,
            "default_currency": "INR",
            "country": "India",
            "create_chart_of_accounts_based_on": "Standard Template",
            "chart_of_accounts": "India - Chart of Accounts",
        }
    ).insert(ignore_permissions=True)
    frappe.defaults.set_global_default("company", COMPANY)
    frappe.defaults.set_global_default("currency", "INR")
    frappe.defaults.set_global_default("country", "India")
    frappe.defaults.set_global_default("time_zone", "Asia/Kolkata")


def create_users():
    frappe.db.set_single_value("System Settings", "allow_login_using_user_name", 1)
    role_names = [
        "Employee",
        "HR User",
        "Projects User",
        "Accounts User",
        "Sales User",
        "Purchase User",
        "Stock User",
    ]
    available_roles = [role for role in role_names if exists("Role", role)]
    exported_by_username = {
        value(row.get("username")): row for row in rows("tabUser") if value(row.get("username"))
    }

    administrator = frappe.get_doc("User", "Administrator")
    administrator.username = "admin"
    administrator.enabled = 1
    administrator.save(ignore_permissions=True)
    update_password("Administrator", credentials[0]["password"])

    expected = {"Administrator", "Guest"}
    for credential in credentials[1:]:
        username = credential["username"]
        email = credential["email"]
        source = exported_by_username.get(username, {})
        expected.add(email)
        if exists("User", email):
            user = frappe.get_doc("User", email)
        else:
            user = frappe.get_doc(
                {
                    "doctype": "User",
                    "email": email,
                    "first_name": value(source.get("first_name"), username.split(".")[0].title()),
                    "last_name": value(source.get("last_name"), username.split(".")[-1].title()),
                    "username": username,
                    "send_welcome_email": 0,
                    "enabled": 1,
                    "user_type": "System User",
                }
            ).insert(ignore_permissions=True)
        user.username = username
        user.enabled = 1
        user.user_type = "System User"
        user.time_zone = "Asia/Kolkata"
        user.set("roles", [{"role": role} for role in available_roles])
        user.save(ignore_permissions=True)
        update_password(email, credential["password"])

    for user_name in frappe.get_all("User", pluck="name"):
        if user_name not in expected:
            frappe.db.set_value("User", user_name, "enabled", 0, update_modified=False)


def create_employees():
    employee_names = {}
    genders = {
        value(source.get("gender"), "Prefer not to say")
        for source in rows("tabEmployee")
    }
    for gender in genders:
        ensure_master("Gender", gender, gender=gender)

    for source in rows("tabEmployee"):
        email = value(source.get("user_id"))
        if not email or not exists("User", email):
            continue
        department = value(source.get("department"))
        designation = value(source.get("designation"))
        if department:
            base_department = department.removesuffix(f" - {COMPANY_ABBR}")
            department = ensure_master(
                "Department",
                department,
                department_name=base_department,
                company=COMPANY,
                is_group=0,
            )
        if designation:
            ensure_master("Designation", designation, designation_name=designation)

        employee_name = value(source.get("name"))
        if exists("Employee", employee_name):
            employee = frappe.get_doc("Employee", employee_name)
        else:
            employee = frappe.get_doc(
                {
                    "doctype": "Employee",
                    "first_name": value(source.get("first_name"), "Employee"),
                    "middle_name": value(source.get("middle_name")),
                    "last_name": value(source.get("last_name")),
                    "user_id": email,
                    "company_email": email,
                    "company": COMPANY,
                    "department": department,
                    "designation": designation,
                    "date_of_joining": value(source.get("date_of_joining"), "2020-01-01"),
                    "date_of_birth": value(source.get("date_of_birth"), "1990-01-01"),
                    "gender": value(source.get("gender"), "Prefer not to say"),
                    "status": "Active",
                }
            )
            employee.insert(ignore_permissions=True, set_name=employee_name)
        employee_names[email] = employee.name
    return employee_names


def create_attendance(employee_names):
    for source in rows("tabAttendance"):
        name = value(source.get("name"))
        if exists("Attendance", name):
            continue
        exported_employee = value(source.get("employee"))
        employee = exported_employee if exists("Employee", exported_employee) else None
        if not employee:
            employee_name = value(source.get("employee_name"))
            employee = frappe.db.get_value("Employee", {"employee_name": employee_name}, "name")
        if not employee:
            continue
        attendance = frappe.get_doc(
            {
                "doctype": "Attendance",
                "employee": employee,
                "attendance_date": value(source.get("attendance_date"), "2026-03-02"),
                "status": value(source.get("status"), "Present"),
                "company": COMPANY,
            }
        )
        attendance.insert(ignore_permissions=True, set_name=name)
        attendance.submit()


def create_business_masters():
    ensure_master(
        "Customer Group",
        "All Customer Groups",
        customer_group_name="All Customer Groups",
        is_group=1,
    )
    ensure_master(
        "Supplier Group",
        "All Supplier Groups",
        supplier_group_name="All Supplier Groups",
        is_group=1,
    )
    ensure_master(
        "Item Group",
        "All Item Groups",
        item_group_name="All Item Groups",
        is_group=1,
    )
    ensure_master(
        "Territory",
        "All Territories",
        territory_name="All Territories",
        is_group=1,
    )
    ensure_master("UOM", "Nos", uom_name="Nos", must_be_whole_number=1)

    ensure_master(
        "Customer Group",
        "Demo Customer Group",
        customer_group_name="Demo Customer Group",
        parent_customer_group="All Customer Groups",
        is_group=0,
    )
    ensure_master(
        "Supplier Group",
        "Demo Supplier Group",
        supplier_group_name="Demo Supplier Group",
        parent_supplier_group="All Supplier Groups",
        is_group=0,
    )
    ensure_master(
        "Item Group",
        "Demo Item Group",
        item_group_name="Demo Item Group",
        parent_item_group="All Item Groups",
        is_group=0,
    )

    for source in rows("tabCustomer"):
        name = value(source.get("name"))
        if not exists("Customer", name):
            frappe.get_doc(
                {
                    "doctype": "Customer",
                    "customer_name": value(source.get("customer_name"), name),
                    "customer_type": value(source.get("customer_type"), "Company"),
                    "customer_group": "Demo Customer Group",
                    "territory": "All Territories",
                }
            ).insert(ignore_permissions=True, set_name=name)

    for source in rows("tabSupplier"):
        name = value(source.get("name"))
        if not exists("Supplier", name):
            frappe.get_doc(
                {
                    "doctype": "Supplier",
                    "supplier_name": value(source.get("supplier_name"), name),
                    "supplier_group": "Demo Supplier Group",
                    "supplier_type": value(source.get("supplier_type"), "Company"),
                    "country": value(source.get("country"), "India"),
                }
            ).insert(ignore_permissions=True, set_name=name)

    for source in rows("tabItem"):
        code = value(source.get("item_code"), value(source.get("name")))
        if not exists("Item", code):
            frappe.get_doc(
                {
                    "doctype": "Item",
                    "item_code": code,
                    "item_name": value(source.get("item_name"), code),
                    "item_group": "Demo Item Group",
                    "stock_uom": value(source.get("stock_uom"), "Nos"),
                    "is_stock_item": int(value(source.get("is_stock_item"), 1)),
                    "standard_rate": float(value(source.get("standard_rate"), 0)),
                }
            ).insert(ignore_permissions=True)

    root_warehouse = f"All Warehouses - {COMPANY_ABBR}"
    if not exists("Warehouse", "Goods In Transit - EAD"):
        frappe.get_doc(
            {
                "doctype": "Warehouse",
                "warehouse_name": "Goods In Transit",
                "company": COMPANY,
                "parent_warehouse": root_warehouse,
                "is_group": 0,
            }
        ).insert(ignore_permissions=True)


def create_projects_and_tasks():
    project_specs = [
        ("Platform Modernization", "Modernize the internal platform and deployment workflow."),
        ("Customer Portal Rollout", "Deliver a self-service portal for enterprise customers."),
        ("Operations Automation", "Automate repeatable finance, HR and support operations."),
    ]
    task_subjects = [
        "Confirm scope and success metrics",
        "Document current-state workflow",
        "Prepare technical design",
        "Create implementation backlog",
        "Build first working increment",
        "Run integration validation",
        "Prepare user acceptance testing",
        "Publish rollout and handover notes",
    ]
    assignees = [credential["email"] for credential in credentials[1:]]
    task_number = 0
    for project_index, (project_name, description) in enumerate(project_specs):
        if not exists("Project", project_name):
            frappe.get_doc(
                {
                    "doctype": "Project",
                    "project_name": project_name,
                    "status": "Open",
                    "priority": "Medium",
                    "company": COMPANY,
                    "expected_start_date": f"2026-09-{1 + project_index * 10:02d}",
                    "expected_end_date": f"2026-{10 + project_index:02d}-15",
                    "notes": description,
                }
            ).insert(ignore_permissions=True, set_name=project_name)
        for offset, subject in enumerate(task_subjects):
            task_name = f"EA-TASK-{project_index + 1:02d}-{offset + 1:02d}"
            if exists("Task", task_name):
                task_number += 1
                continue
            assignee = assignees[task_number % len(assignees)]
            task = frappe.get_doc(
                {
                    "doctype": "Task",
                    "subject": subject,
                    "project": project_name,
                    "status": "Open" if offset < 6 else "Pending Review",
                    "priority": ["High", "Medium", "Low"][offset % 3],
                    "exp_start_date": f"2026-09-{1 + project_index * 10 + offset:02d}",
                    "exp_end_date": f"2026-09-{3 + project_index * 10 + offset:02d}",
                    "description": f"Seeded task for {project_name}. Assigned to {assignee}.",
                }
            )
            task.insert(ignore_permissions=True, set_name=task_name)
            frappe.get_doc(
                {
                    "doctype": "ToDo",
                    "allocated_to": assignee,
                    "reference_type": "Task",
                    "reference_name": task.name,
                    "description": subject,
                    "status": "Open",
                    "priority": task.priority,
                    "date": task.exp_end_date,
                    "assigned_by": "Administrator",
                }
            ).insert(ignore_permissions=True)
            task_number += 1


def main():
    create_company()
    create_users()
    employee_names = create_employees()
    create_business_masters()
    create_attendance(employee_names)
    create_projects_and_tasks()
    frappe.db.commit()
    marker = os.path.join(frappe.get_site_path(), ".enterprise-arena-seeded")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(f"source={os.path.basename(JSON_PATH)}\n")
        handle.write(f"users={len(credentials)}\nprojects=3\ntasks=24\n")
    print(
        "frappe-seed: complete",
        {
            "users": frappe.db.count("User", {"enabled": 1}),
            "employees": frappe.db.count("Employee"),
            "attendance": frappe.db.count("Attendance"),
            "customers": frappe.db.count("Customer"),
            "suppliers": frappe.db.count("Supplier"),
            "items": frappe.db.count("Item", {"item_group": "Demo Item Group"}),
            "projects": frappe.db.count("Project"),
            "tasks": frappe.db.count("Task"),
        },
    )


with open(JSON_PATH, encoding="utf-8") as handle:
    export = json.load(handle)
with open(CREDENTIALS_PATH, encoding="utf-8") as handle:
    credentials = json.load(handle)["users"]

if export.get("service") != "frappe_hrms" or export.get("database") != "mariadb":
    raise RuntimeError("Expected a Frappe HRMS MariaDB JSON export")

frappe.init(site=SITE, sites_path="/home/frappe/frappe-bench/sites")
frappe.connect()
try:
    main()
finally:
    frappe.destroy()
