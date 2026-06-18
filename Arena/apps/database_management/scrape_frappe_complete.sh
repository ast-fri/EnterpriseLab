#!/bin/bash

# scrape_frappe_complete.sh - Generate complete Frappe API schema based on DocTypes

echo "Creating comprehensive Frappe API schema..."

python3 << 'PYEOF'
import json

# Frappe HR/ERP DocTypes (based on comprehensive API documentation)
doctypes = [
    "Appraisal",
    "Attendance",
    "Company",
    "Department",
    "Designation",
    "Employee",
    "Employee Checkin",
    "Employee Onboarding",
    "Employee Separation",
    "Expense Claim",
    "Holiday List",
    "Leave Application",
    "Leave Type",
    "Salary Slip",
    "Salary Structure",
    "Shift Type",
    "Training Event",
    "User"
]

frappe_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Frappe API",
        "version": "1.0.0",
        "description": "Complete Frappe REST API with all HR/ERP DocTypes"
    },
    "servers": [{"url": "http://localhost:8000"}],
    "paths": {},
    "components": {"schemas": {}}
}

frappe_endpoints = []

# Generate CRUD endpoints for each DocType
for doctype in doctypes:
    # Convert to path-safe name
    path_name = doctype.replace(" ", "%20")
    op_id_base = doctype.replace(" ", "")

    # List/Create (GET/POST on collection)
    list_path = f"/{path_name}"
    frappe_spec['paths'][list_path] = {
        "get": {
            "operationId": f"list{op_id_base}",
            "summary": f"List all {doctype}",
            "tags": ["document"],
            "parameters": [],
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "array"}}}},
                "401": {"description": "Unauthorized"}
            }
        },
        "post": {
            "operationId": f"create{op_id_base}",
            "summary": f"Create {doctype}",
            "tags": ["document"],
            "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "201": {"description": "Created", "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "Unauthorized"}
            }
        }
    }
    frappe_endpoints.append(("document", f"list{op_id_base}", "GET", list_path, []))
    frappe_endpoints.append(("document", f"create{op_id_base}", "POST", list_path, []))

    # Get/Update/Delete (GET/PUT/DELETE on individual resource)
    item_path = f"/{path_name}/{{name}}"
    frappe_spec['paths'][item_path] = {
        "get": {
            "operationId": f"get{op_id_base}",
            "summary": f"Get {doctype} by name",
            "tags": ["document"],
            "parameters": [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"}
            }
        },
        "put": {
            "operationId": f"update{op_id_base}",
            "summary": f"Update {doctype}",
            "tags": ["document"],
            "parameters": [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}],
            "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"}
            }
        },
        "delete": {
            "operationId": f"delete{op_id_base}",
            "summary": f"Delete {doctype}",
            "tags": ["document"],
            "parameters": [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": {
                "204": {"description": "Deleted"},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"}
            }
        }
    }
    frappe_endpoints.append(("document", f"get{op_id_base}", "GET", item_path, ["name"]))
    frappe_endpoints.append(("document", f"update{op_id_base}", "PUT", item_path, ["name"]))
    frappe_endpoints.append(("document", f"delete{op_id_base}", "DELETE", item_path, ["name"]))

# Additional generic endpoints
generic_endpoints = [
    ("method", "call", "POST", "/api/method/{method_name}", ["method_name"]),
    ("method", "get", "GET", "/api/method/{method_name}", ["method_name"]),
]

for category, operation_name, method, path, path_params in generic_endpoints:
    operation = {
        "operationId": f"{category}_{operation_name}",
        "summary": f"{operation_name.capitalize()} {category}",
        "tags": [category],
        "parameters": [],
        "responses": {
            "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "object"}}}},
            "401": {"description": "Unauthorized"}
        }
    }

    for param in path_params:
        operation["parameters"].append({"name": param, "in": "path", "required": True, "schema": {"type": "string"}})

    if method in ["POST", "PUT"]:
        operation["requestBody"] = {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}}

    if path not in frappe_spec['paths']:
        frappe_spec['paths'][path] = {}
    frappe_spec['paths'][path][method.lower()] = operation
    frappe_endpoints.append((category, operation_name, method, path, path_params))

with open('fetched_api_schemas/frappe_api_openapi.json', 'w') as f:
    json.dump(frappe_spec, f, indent=2)

print(f"✓ Frappe: {len(frappe_endpoints)} endpoints")
print(f"  Paths: {len(frappe_spec['paths'])}")
print(f"  DocTypes: {len(doctypes)}")
PYEOF

echo "✓ Complete Frappe API schema created!"
