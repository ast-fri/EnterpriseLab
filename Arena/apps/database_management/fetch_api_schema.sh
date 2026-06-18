#!/bin/bash

# fetch_api_schema.sh - Fetch and convert ALL API schemas for all applications
# Main coordinator script that calls the appropriate fetchers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_info "========================================================================="
log_info "Fetching COMPLETE API schemas for all EnterpriseLab applications"
log_info "========================================================================="
echo ""

# Create output directories if they don't exist
mkdir -p "$SCRIPT_DIR/fetched_api_schemas/readable"

# ============================================================================
# STEP 1: Fetch auto-generated OpenAPI specs (Dolibarr, Frappe, Zammad)
#         These apps expose OpenAPI endpoints that we can fetch directly
# ============================================================================

log_info "Step 1: Fetching auto-generated OpenAPI from live applications..."
log_info "        (Dolibarr, Frappe, Zammad)"
echo ""

python3 << 'PYEOF'
import subprocess
import json
import sys

def fetch_dolibarr():
    """Fetch Dolibarr auto-generated OpenAPI"""
    print("  [1/3] Dolibarr...", file=sys.stderr)
    # Dolibarr exposes OpenAPI at /api/index.php/explorer.php
    cmd = ['curl', '-s', 'http://localhost:8082/api/index.php/explorer.php']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout and result.stdout.strip().startswith('{'):
            data = json.loads(result.stdout)
            if 'paths' in data:
                with open('fetched_api_schemas/dolibarr_api_openapi.json', 'w') as f:
                    json.dump(data, f, indent=2)
                return len(data.get('paths', {}))
    except:
        pass
    return 0

def fetch_frappe():
    """Fetch Frappe auto-generated OpenAPI"""
    print("  [2/3] Frappe...", file=sys.stderr)
    # Frappe can generate OpenAPI from DocTypes
    cmd = ['curl', '-s', 'http://localhost:8080/api/method/frappe.core.doctype.doctype.doctype.get_api_spec']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            if 'message' in data and 'paths' in data['message']:
                with open('fetched_api_schemas/frappe_api_openapi.json', 'w') as f:
                    json.dump(data['message'], f, indent=2)
                return len(data['message'].get('paths', {}))
    except:
        pass
    return 0

def fetch_zammad():
    """Fetch Zammad auto-generated OpenAPI"""
    print("  [3/3] Zammad...", file=sys.stderr)
    cmd = ['curl', '-s', 'http://localhost:8080/api/v1/swagger.json']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout and result.stdout.strip().startswith('{'):
            data = json.loads(result.stdout)
            if 'paths' in data:
                with open('fetched_api_schemas/zammad_api_openapi.json', 'w') as f:
                    json.dump(data, f, indent=2)
                return len(data.get('paths', {}))
    except:
        pass
    return 0

# Fetch auto-generated APIs
dolibarr_count = fetch_dolibarr()
frappe_count = fetch_frappe()
zammad_count = fetch_zammad()

print(f"\n  ✓ Dolibarr: {dolibarr_count} paths", file=sys.stderr)
print(f"  ✓ Frappe: {frappe_count} paths", file=sys.stderr)
print(f"  ✓ Zammad: {zammad_count} paths", file=sys.stderr)
PYEOF

echo ""

# ============================================================================
# STEP 2: Generate doc-based OpenAPI specs (RocketChat, Plane, OwnCloud, Dolibarr)
#         These apps don't expose OpenAPI, so we create from documentation
# ============================================================================

log_info "Step 2: Generating OpenAPI from official documentation..."
log_info "        (RocketChat, Plane, OwnCloud, Dolibarr)"
echo ""

"$SCRIPT_DIR/scrape_all_complete_apis.sh"

echo ""

# ============================================================================
# STEP 2.5: Generate comprehensive Frappe API (DocTypes-based)
# ============================================================================

log_info "Step 2.5: Generating comprehensive Frappe API schema..."
echo ""

"$SCRIPT_DIR/scrape_frappe_complete.sh"

echo ""

# ============================================================================
# STEP 2.6: Generate comprehensive Zammad API (Resources-based)
# ============================================================================

log_info "Step 2.6: Generating comprehensive Zammad API schema..."
echo ""

"$SCRIPT_DIR/scrape_zammad_complete.sh"

echo ""

# ============================================================================
# STEP 3: Generate GitLab complete OpenAPI spec
#         GitLab has extensive API that needs separate handling
# ============================================================================

log_info "Step 3: Generating GitLab complete OpenAPI spec..."
echo ""

"$SCRIPT_DIR/scrape_gitlab_complete.sh"

echo ""

# ============================================================================
# STEP 4: Convert all OpenAPI specs to readable format
#         This creates flattened, analyzable JSON for data flow analysis
# ============================================================================

log_info "Step 4: Converting all OpenAPI specs to readable format..."
echo ""

"$SCRIPT_DIR/convert_openapi_to_readable.sh"

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

log_info "========================================================================="
log_info "✓ COMPLETE: API schema fetch and conversion finished!"
log_info "========================================================================="
echo ""

log_info "Files generated:"
echo "  • OpenAPI 3.0 specs: fetched_api_schemas/*_api_openapi.json"
echo "  • Readable format:   fetched_api_schemas/readable/*_api_readable.json"
echo ""

log_info "Endpoint counts per application:"

python3 << 'PYEOF'
import json
import glob
import sys

apps = []
total = 0

for f in sorted(glob.glob("fetched_api_schemas/readable/*_readable.json")):
    try:
        with open(f) as fp:
            data = json.load(fp)
            app = f.split('/')[-1].replace('_api_readable.json', '')
            count = data.get('total_endpoints', 0)
            apps.append((app, count))
            total += count
    except Exception as e:
        print(f"Error reading {f}: {e}", file=sys.stderr)

# Print table
for app, count in apps:
    print(f"  {app:15s} {count:4d} endpoints")

print(f"\n  {'TOTAL':15s} {total:4d} endpoints")
PYEOF

echo ""
log_info "Ready for data flow dependency analysis!"
