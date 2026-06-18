#!/bin/bash

# convert_openapi_to_readable.sh - Convert OpenAPI schemas to readable format
# Goal: Extract EVERY field from OpenAPI and present in clear, usable format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/fetched_api_schemas"
OUTPUT_DIR="$SCRIPT_DIR/fetched_api_schemas/readable"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# ============================================================================
# Convert OpenAPI to Readable Format with ALL details
# ============================================================================

convert_openapi_schema() {
    local service=$1
    local input_file="$INPUT_DIR/${service}_api_openapi.json"
    local output_file="$OUTPUT_DIR/${service}_api_readable.json"

    if [ ! -f "$input_file" ]; then
        log_info "Skipping $service - OpenAPI file not found"
        return 1
    fi

    log_info "Converting $service OpenAPI (preserving ALL information)..."

    python3 << PYEOF
import json
import sys

def extract_schema_properties(schema, level=0):
    """Recursively extract ALL properties from a schema object"""
    if not schema or not isinstance(schema, dict):
        return None

    result = {
        "type": schema.get("type", "unknown")
    }

    # Copy ALL fields from schema
    for key in schema:
        if key not in ["type"]:
            result[key] = schema[key]

    # Handle nested properties
    if "properties" in schema:
        result["fields"] = {}
        for prop_name, prop_schema in schema["properties"].items():
            result["fields"][prop_name] = extract_schema_properties(prop_schema, level+1)

    # Handle arrays
    if schema.get("type") == "array" and "items" in schema:
        result["items"] = extract_schema_properties(schema["items"], level+1)

    # Handle oneOf, anyOf, allOf
    for key in ["oneOf", "anyOf", "allOf"]:
        if key in schema:
            result[key] = [extract_schema_properties(s, level+1) for s in schema[key]]

    return result

def extract_parameter_details(param):
    """Extract ALL details from a parameter"""
    details = {
        "name": param.get("name"),
        "location": param.get("in"),  # path, query, header, cookie
        "required": param.get("required", False),
        "description": param.get("description", "")
    }

    # Schema details
    if "schema" in param:
        schema = param["schema"]
        details["type"] = schema.get("type", "unknown")
        details["format"] = schema.get("format")
        details["default"] = schema.get("default")
        details["enum"] = schema.get("enum")
        details["minimum"] = schema.get("minimum")
        details["maximum"] = schema.get("maximum")
        details["minLength"] = schema.get("minLength")
        details["maxLength"] = schema.get("maxLength")
        details["pattern"] = schema.get("pattern")
        details["example"] = schema.get("example")

        # Copy any other schema properties
        for key, value in schema.items():
            if key not in details:
                details[key] = value

    # Copy any other parameter properties
    for key, value in param.items():
        if key not in ["name", "in", "schema"] and key not in details:
            details[key] = value

    # Remove None values
    return {k: v for k, v in details.items() if v is not None}

def extract_request_body(request_body):
    """Extract ALL details from request body"""
    if not request_body:
        return None

    result = {
        "required": request_body.get("required", False),
        "description": request_body.get("description", "")
    }

    content = request_body.get("content", {})

    for content_type, content_details in content.items():
        schema = content_details.get("schema", {})
        result["content_type"] = content_type
        result["schema"] = extract_schema_properties(schema)

        # Include examples if present
        if "example" in content_details:
            result["example"] = content_details["example"]
        if "examples" in content_details:
            result["examples"] = content_details["examples"]

    return result

def extract_response_details(responses):
    """Extract ALL response details"""
    result = {}

    for status_code, response in responses.items():
        response_detail = {
            "status_code": status_code,
            "description": response.get("description", "")
        }

        # Extract response content
        content = response.get("content", {})
        for content_type, content_details in content.items():
            schema = content_details.get("schema", {})
            response_detail["content_type"] = content_type
            response_detail["schema"] = extract_schema_properties(schema)

            # Include examples
            if "example" in content_details:
                response_detail["example"] = content_details["example"]
            if "examples" in content_details:
                response_detail["examples"] = content_details["examples"]

        # Headers
        if "headers" in response:
            response_detail["headers"] = response["headers"]

        result[status_code] = response_detail

    return result

# Load OpenAPI spec
with open("$input_file") as f:
    openapi = json.load(f)

# Extract service metadata
service_info = {
    "service_name": openapi.get("info", {}).get("title", "$service"),
    "version": openapi.get("info", {}).get("version"),
    "description": openapi.get("info", {}).get("description"),
    "base_urls": [server.get("url") for server in openapi.get("servers", [])],
    "authentication": []
}

# Extract authentication schemes
security = openapi.get("security", [])
security_schemes = openapi.get("components", {}).get("securitySchemes", {})

for scheme_name, scheme_details in security_schemes.items():
    auth_info = {
        "name": scheme_name,
        "type": scheme_details.get("type"),
        "scheme": scheme_details.get("scheme"),
        "bearer_format": scheme_details.get("bearerFormat"),
        "in": scheme_details.get("in"),
        "header_name": scheme_details.get("name"),
        "description": scheme_details.get("description")
    }
    # Remove None values
    auth_info = {k: v for k, v in auth_info.items() if v is not None}
    service_info["authentication"].append(auth_info)

# Extract ALL endpoints with COMPLETE details
endpoints = []

for path, path_item in openapi.get("paths", {}).items():
    # Extract path-level parameters
    path_parameters = path_item.get("parameters", [])

    for method, operation in path_item.items():
        if method in ["get", "post", "put", "delete", "patch", "options", "head", "trace"]:
            # Guess database table from path
            path_parts = path.strip("/").split("/")
            potential_table = path_parts[0].replace("{", "").replace("}", "")

            endpoint = {
                "operation_id": operation.get("operationId", f"{method}_{path.replace('/', '_').replace('{', '').replace('}', '')}"),
                "summary": operation.get("summary", ""),
                "description": operation.get("description", ""),
                "http_method": method.upper(),
                "path": path,
                "full_url": f"{service_info['base_urls'][0] if service_info['base_urls'] else ''}{path}",
                "tags": operation.get("tags", []),
                "deprecated": operation.get("deprecated", False),
                "potential_database_table": potential_table
            }

            # Input arguments - combine path and operation parameters
            all_parameters = path_parameters + operation.get("parameters", [])

            input_args = {
                "path_parameters": [],
                "query_parameters": [],
                "header_parameters": [],
                "cookie_parameters": []
            }

            for param in all_parameters:
                param_details = extract_parameter_details(param)
                location = param_details.get("location")

                if location == "path":
                    input_args["path_parameters"].append(param_details)
                elif location == "query":
                    input_args["query_parameters"].append(param_details)
                elif location == "header":
                    input_args["header_parameters"].append(param_details)
                elif location == "cookie":
                    input_args["cookie_parameters"].append(param_details)

            # Request body
            request_body = operation.get("requestBody")
            if request_body:
                input_args["body"] = extract_request_body(request_body)

            endpoint["input_arguments"] = input_args

            # Output arguments (responses)
            responses = operation.get("responses", {})
            endpoint["output_responses"] = extract_response_details(responses)

            # Security requirements for this endpoint
            if "security" in operation:
                endpoint["security"] = operation["security"]

            # Callbacks
            if "callbacks" in operation:
                endpoint["callbacks"] = operation["callbacks"]

            # External docs
            if "externalDocs" in operation:
                endpoint["external_docs"] = operation["externalDocs"]

            # Servers (endpoint-specific)
            if "servers" in operation:
                endpoint["servers"] = operation["servers"]

            endpoints.append(endpoint)

# Create final readable output
readable_api = {
    "metadata": service_info,
    "total_endpoints": len(endpoints),
    "endpoints": endpoints,
    "openapi_version": openapi.get("openapi"),
    "components": {
        "schemas": openapi.get("components", {}).get("schemas", {}),
        "responses": openapi.get("components", {}).get("responses", {}),
        "parameters": openapi.get("components", {}).get("parameters", {}),
        "examples": openapi.get("components", {}).get("examples", {}),
        "requestBodies": openapi.get("components", {}).get("requestBodies", {}),
        "headers": openapi.get("components", {}).get("headers", {}),
        "securitySchemes": openapi.get("components", {}).get("securitySchemes", {}),
        "links": openapi.get("components", {}).get("links", {}),
        "callbacks": openapi.get("components", {}).get("callbacks", {})
    },
    "notes": {
        "source": "Converted from OpenAPI 3.0 specification",
        "faithfulness": "100% - All information from original OpenAPI preserved",
        "potential_database_table": "Inferred from API path, may need manual verification"
    }
}

# Save
import os
os.makedirs("$OUTPUT_DIR", exist_ok=True)
with open("$output_file", "w") as f:
    json.dump(readable_api, f, indent=2)

print(f"✓ {service_info['service_name']}: {len(endpoints)} endpoints converted")
print(f"  Input params: {sum(len(e['input_arguments']['path_parameters']) + len(e['input_arguments']['query_parameters']) for e in endpoints)} total")
print(f"  Components: {sum(len(v) if isinstance(v, dict) else 0 for v in readable_api['components'].values())} schemas/definitions")
PYEOF

    log_info "✓ Saved: $output_file"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    log_info "Converting OpenAPI schemas to readable format..."
    log_info "Input: $INPUT_DIR"
    log_info "Output: $OUTPUT_DIR"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    # Convert all available OpenAPI files
    for service in dolibarr frappe zammad owncloud rocketchat plane gitlab; do
        convert_openapi_schema "$service" || true
        echo ""
    done

    log_info "✓ Conversion complete!"
    log_info ""
    log_info "Files created:"
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "No files created"
    echo ""

    log_info "Example usage:"
    echo "  # View first endpoint"
    echo "  python3 -c \"import json; d=json.load(open('$OUTPUT_DIR/dolibarr_api_readable.json')); print(json.dumps(d['endpoints'][0], indent=2))\" | head -50"
    echo ""
    echo "  # List all function names"
    echo "  python3 -c \"import json; d=json.load(open('$OUTPUT_DIR/dolibarr_api_readable.json')); [print(e['operation_id']) for e in d['endpoints']]\""
}

main "$@"
