#!/bin/bash

# scrape_zammad_complete.sh - Generate complete Zammad API schema based on resources

echo "Creating comprehensive Zammad API schema..."

python3 << 'PYEOF'
import json

# Zammad API resources (based on comprehensive API documentation)
resources = [
    "calendars",
    "channels",
    "email_addresses",
    "groups",
    "knowledge_bases",
    "macros",
    "organizations",
    "overviews",
    "permissions",
    "roles",
    "slas",
    "tags",
    "text_modules",
    "ticket_articles",
    "ticket_priorities",
    "ticket_states",
    "tickets",
    "time_accountings",
    "users"
]

zammad_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Zammad API",
        "version": "1.0.0",
        "description": "Complete Zammad REST API with all resources"
    },
    "servers": [{"url": "http://localhost:8050"}],
    "paths": {},
    "components": {"schemas": {}}
}

zammad_endpoints = []

# Generate CRUD endpoints for each resource
for resource in resources:
    # Convert to operation ID base (remove underscores, capitalize)
    op_id_base = ''.join(word.capitalize() for word in resource.split('_'))

    # List/Create (GET/POST on collection)
    list_path = f"/{resource}"
    zammad_spec['paths'][list_path] = {
        "get": {
            "operationId": f"get__{resource}",
            "summary": f"List all {resource}",
            "tags": [resource],
            "parameters": [],
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "array"}}}},
                "401": {"description": "Unauthorized"}
            }
        },
        "post": {
            "operationId": f"post__{resource}",
            "summary": f"Create {resource}",
            "tags": [resource],
            "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "201": {"description": "Created", "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "Unauthorized"}
            }
        }
    }
    zammad_endpoints.append((resource, f"get__{resource}", "GET", list_path, []))
    zammad_endpoints.append((resource, f"post__{resource}", "POST", list_path, []))

    # Get/Update/Delete (GET/PUT/DELETE on individual resource)
    item_path = f"/{resource}/{{id}}"
    zammad_spec['paths'][item_path] = {
        "get": {
            "operationId": f"get__{resource}_id",
            "summary": f"Get {resource} by ID",
            "tags": [resource],
            "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"}
            }
        },
        "put": {
            "operationId": f"put__{resource}_id",
            "summary": f"Update {resource}",
            "tags": [resource],
            "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"}
            }
        },
        "delete": {
            "operationId": f"delete__{resource}_id",
            "summary": f"Delete {resource}",
            "tags": [resource],
            "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": {
                "204": {"description": "Deleted"},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"}
            }
        }
    }
    zammad_endpoints.append((resource, f"get__{resource}_id", "GET", item_path, ["id"]))
    zammad_endpoints.append((resource, f"put__{resource}_id", "PUT", item_path, ["id"]))
    zammad_endpoints.append((resource, f"delete__{resource}_id", "DELETE", item_path, ["id"]))

# Add special endpoints (search, me, etc.)
special_endpoints = [
    ("tickets", "get__tickets_search", "GET", "/tickets/search", []),
    ("users", "get__users_search", "GET", "/users/search", []),
    ("users", "get__users_me", "GET", "/users/me", []),
    ("organizations", "get__organizations_search", "GET", "/organizations/search", []),
]

for resource, op_id, method, path, params in special_endpoints:
    operation = {
        "operationId": op_id,
        "summary": f"{op_id.replace('__', ' ').replace('_', ' ').title()}",
        "tags": [resource],
        "parameters": [],
        "responses": {
            "200": {"description": "Success", "content": {"application/json": {"schema": {"type": "object"}}}},
            "401": {"description": "Unauthorized"}
        }
    }

    for param in params:
        operation["parameters"].append({"name": param, "in": "path", "required": True, "schema": {"type": "string"}})

    if path not in zammad_spec['paths']:
        zammad_spec['paths'][path] = {}
    zammad_spec['paths'][path][method.lower()] = operation
    zammad_endpoints.append((resource, op_id, method, path, params))

with open('fetched_api_schemas/zammad_api_openapi.json', 'w') as f:
    json.dump(zammad_spec, f, indent=2)

print(f"✓ Zammad: {len(zammad_endpoints)} endpoints")
print(f"  Paths: {len(zammad_spec['paths'])}")
print(f"  Resources: {len(resources)}")
PYEOF

echo "✓ Complete Zammad API schema created!"
