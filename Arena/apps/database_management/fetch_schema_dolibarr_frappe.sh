#!/bin/bash

# Special script for Dolibarr and Frappe schemas (due to large table count)
# Uses Python to build the nested structure instead of complex SQL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/fetched_schemas"

GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# ============================================================================
# DOLIBARR SCHEMA
# ============================================================================
log_info "Fetching Dolibarr schema..."

python3 << 'PYEOF'
import json
import subprocess

DB_CONTAINER = "dolibarr-mariadb-1"
DB_NAME = "dolidb"
DB_USER = "root"
DB_PASS = "root"

# Get all tables
tables_cmd = f"docker exec {DB_CONTAINER} mariadb -u{DB_USER} -p{DB_PASS} {DB_NAME} -se \"SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = '{DB_NAME}' ORDER BY TABLE_NAME;\""
result = subprocess.run(tables_cmd, shell=True, capture_output=True, text=True)
table_names = result.stdout.strip().split('\n') if result.stdout else []

tables = []
for table_name in table_names[:50]:  # Limit to first 50 tables for performance
    # Get columns for this table
    cols_cmd = f"""docker exec {DB_CONTAINER} mariadb -u{DB_USER} -p{DB_PASS} {DB_NAME} -se "
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'column_name', COLUMN_NAME,
                'ordinal_position', ORDINAL_POSITION,
                'column_default', COLUMN_DEFAULT,
                'is_nullable', IS_NULLABLE,
                'data_type', DATA_TYPE,
                'character_maximum_length', CHARACTER_MAXIMUM_LENGTH,
                'numeric_precision', NUMERIC_PRECISION,
                'numeric_scale', NUMERIC_SCALE,
                'column_type', COLUMN_TYPE,
                'column_key', COLUMN_KEY,
                'extra', EXTRA,
                'column_comment', COLUMN_COMMENT
            )
        )
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = '{DB_NAME}' AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION;
    " """

    result = subprocess.run(cols_cmd, shell=True, capture_output=True, text=True)
    try:
        columns = json.loads(result.stdout.strip()) if result.stdout.strip() else []
        tables.append({
            'name': table_name,
            'columns': columns if columns else []
        })
    except:
        tables.append({
            'name': table_name,
            'columns': []
        })

output = {
    "service": "dolibarr",
    "database_type": "mariadb",
    "database_name": DB_NAME,
    "url": "http://localhost:8082",
    "tables": tables,
    "timestamp": subprocess.run('date -Iseconds', shell=True, capture_output=True, text=True).stdout.strip(),
    "note": "Limited to first 50 tables for performance. Use information_schema queries for complete schema."
}

with open('fetched_schemas/dolibarr_schema.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Dolibarr: {len(tables)} tables extracted")
PYEOF

log_info "Dolibarr schema saved"

# ============================================================================
# FRAPPE SCHEMA
# ============================================================================
log_info "Fetching Frappe schema..."

python3 << 'PYEOF'
import json
import subprocess

DB_CONTAINER = "docker-mariadb-1"
DB_USER = "root"
DB_PASS = "123"

# Get Frappe database
dbs_cmd = f"docker exec {DB_CONTAINER} mariadb -u{DB_USER} -p{DB_PASS} -e \"SHOW DATABASES;\" 2>/dev/null | grep '^_'"
result = subprocess.run(dbs_cmd, shell=True, capture_output=True, text=True)
frappe_dbs = result.stdout.strip().split('\n') if result.stdout else []

if not frappe_dbs:
    print("No Frappe databases found")
    exit(1)

# Use first database
SITE_DB = frappe_dbs[0].strip()

# Get all tab* tables
tables_cmd = f"docker exec {DB_CONTAINER} mariadb -u{DB_USER} -p{DB_PASS} {SITE_DB} -se \"SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = '{SITE_DB}' AND TABLE_NAME LIKE 'tab%' ORDER BY TABLE_NAME;\""
result = subprocess.run(tables_cmd, shell=True, capture_output=True, text=True)
table_names = result.stdout.strip().split('\n') if result.stdout else []

tables = []
for i, table_name in enumerate(table_names):
    if i >= 100:  # Limit to first 100 DocTypes for performance
        break

    # Get columns for this table
    cols_cmd = f"""docker exec {DB_CONTAINER} mariadb -u{DB_USER} -p{DB_PASS} {SITE_DB} -se "
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'column_name', COLUMN_NAME,
                'ordinal_position', ORDINAL_POSITION,
                'column_default', COLUMN_DEFAULT,
                'is_nullable', IS_NULLABLE,
                'data_type', DATA_TYPE,
                'character_maximum_length', CHARACTER_MAXIMUM_LENGTH,
                'numeric_precision', NUMERIC_PRECISION,
                'numeric_scale', NUMERIC_SCALE,
                'column_type', COLUMN_TYPE,
                'column_key', COLUMN_KEY,
                'extra', EXTRA,
                'column_comment', COLUMN_COMMENT
            )
        )
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = '{SITE_DB}' AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION;
    " """

    result = subprocess.run(cols_cmd, shell=True, capture_output=True, text=True)
    try:
        columns = json.loads(result.stdout.strip()) if result.stdout.strip() else []
        tables.append({
            'name': table_name,
            'columns': columns if columns else []
        })
    except:
        tables.append({
            'name': table_name,
            'columns': []
        })

output = {
    "service": "frappe_hrms",
    "database_type": "mariadb",
    "database_name": SITE_DB,
    "url": "http://localhost:8084",
    "tables": tables,
    "timestamp": subprocess.run('date -Iseconds', shell=True, capture_output=True, text=True).stdout.strip(),
    "note": "Limited to first 100 DocTypes (tables starting with 'tab') for performance."
}

with open('fetched_schemas/frappe_schema.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Frappe: {len(tables)} DocTypes extracted")
PYEOF

log_info "Frappe schema saved"
log_info "✓ Done!"
PYEOF
