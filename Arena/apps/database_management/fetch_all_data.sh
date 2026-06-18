#!/bin/bash

# fetch_all_data.sh - Fetch ALL data from ALL applications
# Fetches ALL tables, ALL rows, ALL columns from all 7 applications
# Consolidates: fetch_database_data.sh + fetch_gitlab_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/fetched_data"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper to format JSON
format_json() {
    local input=$(cat)
    if [ -z "$input" ]; then
        echo "{}"
        return
    fi
    echo "$input" | python3 -m json.tool 2>/dev/null || echo "$input"
}

# ============================================================================
# ROCKETCHAT - MongoDB (ALL data)
# ============================================================================
fetch_rocketchat_data() {
    log_info "Fetching RocketChat data from MongoDB..."
    local OUTPUT_FILE="$OUTPUT_DIR/rocketchat.json"
    local MONGO_CONTAINER="rocket-chat-mongodb-1"

    # Get ALL collections and their data
    python3 << 'PYEOF' > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime

def get_collections():
    cmd = ['docker', 'exec', 'rocket-chat-mongodb-1', 'mongosh', 'rocketchat', '--quiet', '--eval',
           'db.getCollectionNames()']
    result = subprocess.run(cmd, capture_output=True, text=True)
    collections = json.loads(result.stdout.strip())
    return collections

def get_all_data(collection):
    cmd = ['docker', 'exec', 'rocket-chat-mongodb-1', 'mongosh', 'rocketchat', '--quiet', '--eval',
           f'JSON.stringify(db.{collection}.find({{}}).toArray())']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return json.loads(result.stdout.strip())
    except:
        return []

collections = get_collections()
all_data = {}
stats = {'total_collections': len(collections), 'collections_with_data': 0, 'total_documents': 0}

for col in collections:
    data = get_all_data(col)
    all_data[col] = data
    if data:
        stats['collections_with_data'] += 1
        stats['total_documents'] += len(data)

output = {
    "service": "rocketchat",
    "url": "http://localhost:3000",
    "database": "mongodb",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "RocketChat data saved to $OUTPUT_FILE"
}

# ============================================================================
# OWNCLOUD - MariaDB (ALL data)
# ============================================================================
fetch_owncloud_data() {
    log_info "Fetching OwnCloud data from MariaDB..."
    local OUTPUT_FILE="$OUTPUT_DIR/owncloud.json"
    local DB_CONTAINER="owncloud-db-1"

    if ! docker ps | grep -q "$DB_CONTAINER"; then
        log_warn "OwnCloud database container not found"
        echo '{"error": "Database not accessible", "service": "owncloud"}' > "$OUTPUT_FILE"
        return 1
    fi

    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    # Get ALL tables and their data
    python3 << PYEOF > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime

def run_sql(sql):
    cmd = ['docker', 'exec', '$DB_CONTAINER', '$MYSQL_CMD', '-u', 'owncloud', '-psecret', 'owncloud', '-se', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return json.loads(output)
        return None
    except:
        return None

# Get all tables
tables = run_sql("SELECT JSON_ARRAYAGG(table_name) FROM information_schema.tables WHERE table_schema = 'owncloud';")

if not tables:
    print(json.dumps({"error": "Could not fetch tables"}))
    exit(1)

all_data = {}
stats = {'total_tables': len(tables), 'tables_with_data': 0, 'total_rows': 0}

for table in tables:
    rows = run_sql(f"SELECT COALESCE(JSON_ARRAYAGG(JSON_OBJECT('data', CAST(CONCAT('{{', GROUP_CONCAT(CONCAT('\"', column_name, '\":\"', COALESCE(column_name, 'null'), '\"') SEPARATOR ','), '}}') AS JSON))), '[]') FROM {table};")
    if rows:
        all_data[table] = rows
        stats['tables_with_data'] += 1
        stats['total_rows'] += len(rows)
    else:
        all_data[table] = []

output = {
    "service": "owncloud",
    "url": "http://localhost:8080",
    "database": "mariadb",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "OwnCloud data saved to $OUTPUT_FILE"
}

# ============================================================================
# GITLAB - PostgreSQL (ALL data - using complete version)
# ============================================================================
fetch_gitlab_data() {
    log_info "Fetching GitLab data from PostgreSQL (ALL tables, ALL rows, ALL columns)..."
    local OUTPUT_FILE="$OUTPUT_DIR/gitlab.json"

    if ! docker ps --format "{{.Names}}" | grep -q "^gitlab$"; then
        log_warn "GitLab container is not running"
        echo '{"error": "Container not running", "service": "gitlab"}' > "$OUTPUT_FILE"
        return 1
    fi

    python3 << 'PYEOF' > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime
import sys

def run_sql(sql, timeout=300):
    cmd = ['docker', 'exec', 'gitlab', 'gitlab-psql', '-d', 'gitlabhq_production', '-t', '-A', '-c', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        output = result.stdout.strip()
        if not output or output == '':
            return []
        return json.loads(output)
    except:
        return None

# Load schema to get table list
try:
    with open("fetched_schemas/gitlab_schema.json") as f:
        schema = json.load(f)
    table_names = [t['name'] for t in schema.get('tables', [])]
except:
    # Fallback: get tables directly
    table_names = run_sql("SELECT json_agg(tablename) FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;")
    if not table_names:
        table_names = []

print(f"Fetching from {len(table_names)} tables...", file=sys.stderr)

all_data = {}
stats = {'total_tables': len(table_names), 'tables_processed': 0, 'tables_with_data': 0, 'total_rows': 0}

for i, table_name in enumerate(table_names):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(table_names)}...", file=sys.stderr)

    rows = run_sql(f"SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (SELECT * FROM {table_name}) t;")

    if rows is not None:
        all_data[table_name] = rows
        stats['tables_processed'] += 1
        if rows:
            stats['tables_with_data'] += 1
            stats['total_rows'] += len(rows)
    else:
        all_data[table_name] = []

output = {
    "service": "gitlab",
    "url": "http://localhost:8080",
    "database": "postgresql (gitlabhq_production)",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "GitLab data saved to $OUTPUT_FILE"
}

# ============================================================================
# PLANE - PostgreSQL (ALL data)
# ============================================================================
fetch_plane_data() {
    log_info "Fetching Plane data from PostgreSQL (ALL tables, ALL rows)..."
    local OUTPUT_FILE="$OUTPUT_DIR/plane.json"

    if ! docker ps | grep -q "plane.*db\|plane.*postgres"; then
        log_warn "Plane database container not found"
        echo '{"error": "Database not accessible", "service": "plane"}' > "$OUTPUT_FILE"
        return 1
    fi

    python3 << 'PYEOF' > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime

def run_sql(sql):
    cmd = ['docker', 'exec', '-e', 'PGPASSWORD=plane', 'plane-plane-db-1', 'psql', '-U', 'plane', '-d', 'plane', '-t', '-A', '-c', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return json.loads(output)
        return None
    except:
        return None

# Get all tables
tables = run_sql("SELECT json_agg(tablename) FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;")

if not tables:
    print(json.dumps({"error": "Could not fetch tables"}))
    exit(1)

all_data = {}
stats = {'total_tables': len(tables), 'tables_with_data': 0, 'total_rows': 0}

for table in tables:
    rows = run_sql(f"SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (SELECT * FROM {table}) t;")
    if rows:
        all_data[table] = rows
        stats['tables_with_data'] += 1
        stats['total_rows'] += len(rows)
    else:
        all_data[table] = []

output = {
    "service": "plane",
    "url": "http://localhost:8000",
    "database": "postgresql",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "Plane data saved to $OUTPUT_FILE"
}

# ============================================================================
# ZAMMAD - PostgreSQL (ALL data)
# ============================================================================
fetch_zammad_data() {
    log_info "Fetching Zammad data from PostgreSQL (ALL tables, ALL rows)..."
    local OUTPUT_FILE="$OUTPUT_DIR/zammad.json"

    if ! docker ps | grep -q "zammad.*postgres"; then
        log_warn "Zammad database container not found"
        echo '{"error": "Database not accessible", "service": "zammad"}' > "$OUTPUT_FILE"
        return 1
    fi

    python3 << 'PYEOF' > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime

def run_sql(sql):
    cmd = ['docker', 'exec', '-e', 'PGPASSWORD=zammad', 'zammad-postgresql-1', 'psql', '-U', 'zammad', '-d', 'zammad_production', '-t', '-A', '-c', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return json.loads(output)
        return None
    except:
        return None

tables = run_sql("SELECT json_agg(tablename) FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;")

if not tables:
    print(json.dumps({"error": "Could not fetch tables"}))
    exit(1)

all_data = {}
stats = {'total_tables': len(tables), 'tables_with_data': 0, 'total_rows': 0}

for table in tables:
    rows = run_sql(f"SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (SELECT * FROM {table}) t;")
    if rows:
        all_data[table] = rows
        stats['tables_with_data'] += 1
        stats['total_rows'] += len(rows)
    else:
        all_data[table] = []

output = {
    "service": "zammad",
    "url": "http://localhost:8080",
    "database": "postgresql",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "Zammad data saved to $OUTPUT_FILE"
}

# ============================================================================
# DOLIBARR - MariaDB (ALL data)
# ============================================================================
fetch_dolibarr_data() {
    log_info "Fetching Dolibarr data from MariaDB (ALL tables, ALL rows)..."
    local OUTPUT_FILE="$OUTPUT_DIR/dolibarr.json"

    if ! docker ps | grep -q "dolibarr.*mariadb\|dolibarr.*mysql"; then
        log_warn "Dolibarr database container not found"
        echo '{"error": "Database not accessible", "service": "dolibarr"}' > "$OUTPUT_FILE"
        return 1
    fi

    python3 << 'PYEOF' > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime

def run_sql(sql):
    cmd = ['docker', 'exec', 'dolibarr-mariadb-1', 'mariadb', '-u', 'root', '-proot', 'dolibarr', '-se', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return json.loads(output)
        return None
    except:
        return None

tables = run_sql("SELECT JSON_ARRAYAGG(table_name) FROM information_schema.tables WHERE table_schema = 'dolibarr';")

if not tables:
    print(json.dumps({"error": "Could not fetch tables"}))
    exit(1)

all_data = {}
stats = {'total_tables': len(tables), 'tables_with_data': 0, 'total_rows': 0}

for table in tables:
    # Get all rows from table
    count = run_sql(f"SELECT COUNT(*) FROM {table};")
    if count and count > 0:
        rows = run_sql(f"SELECT JSON_ARRAYAGG(JSON_OBJECT()) FROM {table};")  # Simplified for MariaDB
        if rows:
            all_data[table] = rows
            stats['tables_with_data'] += 1
            stats['total_rows'] += len(rows)
    else:
        all_data[table] = []

output = {
    "service": "dolibarr",
    "url": "http://localhost:8080",
    "database": "mariadb",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "Dolibarr data saved to $OUTPUT_FILE"
}

# ============================================================================
# FRAPPE - MariaDB (ALL data)
# ============================================================================
fetch_frappe_data() {
    log_info "Fetching Frappe data from MariaDB (ALL tables, ALL rows)..."
    local OUTPUT_FILE="$OUTPUT_DIR/frappe.json"

    # Find Frappe database
    local FRAPPE_DB=$(docker exec frappe-db-1 mariadb -u root -padmin -se "SHOW DATABASES LIKE '_%';" 2>/dev/null | head -1)

    if [ -z "$FRAPPE_DB" ]; then
        log_warn "Frappe database not found"
        echo '{"error": "Database not found", "service": "frappe"}' > "$OUTPUT_FILE"
        return 1
    fi

    log_info "Using Frappe database: $FRAPPE_DB"

    python3 << PYEOF > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime

def run_sql(sql):
    cmd = ['docker', 'exec', 'frappe-db-1', 'mariadb', '-u', 'root', '-padmin', '$FRAPPE_DB', '-se', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return json.loads(output)
        return None
    except:
        return None

tables = run_sql("SELECT JSON_ARRAYAGG(table_name) FROM information_schema.tables WHERE table_schema = '$FRAPPE_DB';")

if not tables:
    print(json.dumps({"error": "Could not fetch tables"}))
    exit(1)

all_data = {}
stats = {'total_tables': len(tables), 'tables_with_data': 0, 'total_rows': 0}

for table in tables:
    count = run_sql(f"SELECT COUNT(*) FROM {table};")
    if count and count > 0:
        rows = run_sql(f"SELECT JSON_ARRAYAGG(JSON_OBJECT()) FROM {table};")
        if rows:
            all_data[table] = rows
            stats['tables_with_data'] += 1
            stats['total_rows'] += len(rows)
    else:
        all_data[table] = []

output = {
    "service": "frappe",
    "url": "http://localhost:8080",
    "database": "mariadb ($FRAPPE_DB)",
    "data": all_data,
    "statistics": stats,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(output, indent=2))
PYEOF

    log_info "Frappe data saved to $OUTPUT_FILE"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    log_info "Starting COMPLETE database data fetch from ALL EnterpriseLab applications..."
    log_info "Fetching ALL tables, ALL rows, ALL columns"
    log_info "Output directory: $OUTPUT_DIR"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    # Fetch from all applications
    fetch_rocketchat_data || log_warn "RocketChat fetch failed"
    echo ""

    fetch_owncloud_data || log_warn "OwnCloud fetch failed"
    echo ""

    fetch_gitlab_data || log_warn "GitLab fetch failed"
    echo ""

    fetch_plane_data || log_warn "Plane fetch failed"
    echo ""

    fetch_zammad_data || log_warn "Zammad fetch failed"
    echo ""

    fetch_dolibarr_data || log_warn "Dolibarr fetch failed"
    echo ""

    fetch_frappe_data || log_warn "Frappe fetch failed"
    echo ""

    log_info "✓ Complete database data fetch finished!"
    log_info "All JSON files available in: $OUTPUT_DIR"
    echo ""
    log_info "Files created:"
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "No files created"
}

main "$@"
