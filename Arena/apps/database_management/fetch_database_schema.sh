#!/bin/bash

# fetch_database_schema.sh - Extract database schema (tables and columns) from applications
# This script extracts the complete schema structure of all databases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/fetched_schemas"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper function to format JSON
format_json() {
    python3 -m json.tool 2>/dev/null || cat
}

# ============================================================================
# ROCKETCHAT - Team communication platform (MongoDB)
# ============================================================================
fetch_rocketchat_schema() {
    log_info "Fetching RocketChat MongoDB schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/rocketchat_schema.json"
    local MONGO_CONTAINER="rocket-chat-mongodb-1"

    if ! docker ps | grep -q "$MONGO_CONTAINER"; then
        log_warn "RocketChat MongoDB container not found"
        echo '{"error": "Container not running", "service": "rocketchat"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Get all collections
    local COLLECTIONS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.getCollectionNames())
    ' 2>/dev/null || echo '[]')

    # Get schema for all collections in tables format
    local SCHEMAS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval "
        const collections = db.getCollectionNames();
        const tables = [];

        collections.forEach(collName => {
            const sample = db.getCollection(collName).findOne();
            const columns = [];

            if (sample) {
                Object.keys(sample).forEach((key, index) => {
                    const value = sample[key];
                    let dataType = typeof value;

                    if (Array.isArray(value)) {
                        dataType = 'array';
                        if (value.length > 0) {
                            dataType = 'array<' + typeof value[0] + '>';
                        }
                    } else if (value instanceof Date) {
                        dataType = 'date';
                    } else if (value === null) {
                        dataType = 'null';
                    } else if (value && typeof value === 'object' && value.constructor.name === 'ObjectId') {
                        dataType = 'ObjectId';
                    } else if (value && typeof value === 'object') {
                        dataType = 'object';
                    }

                    columns.push({
                        column_name: key,
                        ordinal_position: index + 1,
                        data_type: dataType,
                        is_nullable: value === null || value === undefined ? 'YES' : 'NO',
                        sample_value: value === null ? null : (
                            typeof value === 'string' ? value.substring(0, 100) :
                            typeof value === 'object' ? JSON.stringify(value).substring(0, 100) :
                            value
                        )
                    });
                });
            }

            tables.push({
                name: collName,
                columns: columns,
                document_count: db.getCollection(collName).countDocuments()
            });
        });

        JSON.stringify(tables);
    " 2>/dev/null || echo '[]')

    echo "{
        \"service\": \"rocketchat\",
        \"database_type\": \"mongodb\",
        \"url\": \"http://localhost:3000\",
        \"tables\": $SCHEMAS,
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "RocketChat schema saved to $OUTPUT_FILE"
}

# ============================================================================
# OWNCLOUD - File sharing (MariaDB)
# ============================================================================
fetch_owncloud_schema() {
    log_info "Fetching OwnCloud MariaDB schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/owncloud_schema.json"
    local DB_CONTAINER="owncloud-db-1"

    if ! docker ps | grep -q "$DB_CONTAINER"; then
        log_warn "OwnCloud database container not found"
        echo '{"error": "Container not running", "service": "owncloud"}' > "$OUTPUT_FILE"
        return 1
    fi

    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    # Get all tables
    local TABLES=$(docker exec $DB_CONTAINER $MYSQL_CMD -uowncloud -psecret owncloud -se "
        SELECT JSON_ARRAYAGG(TABLE_NAME)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = 'owncloud';
    " 2>/dev/null || echo '[]')

    # Get detailed schema grouped by table
    local SCHEMA=$(docker exec $DB_CONTAINER $MYSQL_CMD -uowncloud -psecret owncloud -se "
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', TABLE_NAME,
                'columns', (
                    SELECT JSON_ARRAYAGG(
                        JSON_OBJECT(
                            'column_name', COLUMN_NAME,
                            'ordinal_position', ORDINAL_POSITION,
                            'column_default', COLUMN_DEFAULT,
                            'is_nullable', IS_NULLABLE,
                            'data_type', DATA_TYPE,
                            'character_maximum_length', CHARACTER_MAXIMUM_LENGTH,
                            'character_octet_length', CHARACTER_OCTET_LENGTH,
                            'numeric_precision', NUMERIC_PRECISION,
                            'numeric_scale', NUMERIC_SCALE,
                            'datetime_precision', DATETIME_PRECISION,
                            'character_set_name', CHARACTER_SET_NAME,
                            'collation_name', COLLATION_NAME,
                            'column_type', COLUMN_TYPE,
                            'column_key', COLUMN_KEY,
                            'extra', EXTRA,
                            'privileges', PRIVILEGES,
                            'column_comment', COLUMN_COMMENT
                        )
                        ORDER BY ORDINAL_POSITION
                    )
                    FROM information_schema.COLUMNS c2
                    WHERE c2.TABLE_SCHEMA = 'owncloud'
                    AND c2.TABLE_NAME = t.TABLE_NAME
                )
            )
        )
        FROM (SELECT DISTINCT TABLE_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = 'owncloud') t;
    " 2>/dev/null || echo '[]')

    # Get row counts per table
    local COUNTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -uowncloud -psecret owncloud -se "
        SELECT JSON_OBJECTAGG(TABLE_NAME, TABLE_ROWS)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = 'owncloud';
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"owncloud\",
        \"database_type\": \"mariadb\",
        \"database_name\": \"owncloud\",
        \"url\": \"http://localhost:8081\",
        \"tables\": $SCHEMA,
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "OwnCloud schema saved to $OUTPUT_FILE"
}

# ============================================================================
# PLANE - Project management (PostgreSQL)
# ============================================================================
fetch_plane_schema() {
    log_info "Fetching Plane PostgreSQL schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/plane_schema.json"
    local DB_CONTAINER="plane-plane-db-1"

    if ! docker ps | grep -q "$DB_CONTAINER"; then
        log_warn "Plane database container not found"
        echo '{"error": "Container not running", "service": "plane"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Get all tables
    local TABLES=$(docker exec -e PGPASSWORD=plane $DB_CONTAINER psql -U plane -d plane -t -c "
        SELECT json_agg(tablename)
        FROM pg_tables
        WHERE schemaname = 'public';
    " 2>/dev/null || echo '[]')

    # Get detailed schema grouped by table
    local SCHEMA=$(docker exec -e PGPASSWORD=plane $DB_CONTAINER psql -U plane -d plane -t -c "
        SELECT json_agg(
            json_build_object(
                'name', t.table_name,
                'columns', (
                    SELECT json_agg(
                        json_build_object(
                            'column_name', c.column_name,
                            'ordinal_position', c.ordinal_position,
                            'column_default', c.column_default,
                            'is_nullable', c.is_nullable,
                            'data_type', c.data_type,
                            'character_maximum_length', c.character_maximum_length,
                            'character_octet_length', c.character_octet_length,
                            'numeric_precision', c.numeric_precision,
                            'numeric_precision_radix', c.numeric_precision_radix,
                            'numeric_scale', c.numeric_scale,
                            'datetime_precision', c.datetime_precision,
                            'character_set_name', c.character_set_name,
                            'collation_name', c.collation_name,
                            'udt_name', c.udt_name,
                            'is_updatable', c.is_updatable
                        )
                        ORDER BY c.ordinal_position
                    )
                    FROM information_schema.columns c
                    WHERE c.table_schema = 'public'
                    AND c.table_name = t.table_name
                )
            )
        )
        FROM (SELECT DISTINCT table_name FROM information_schema.columns WHERE table_schema = 'public') t;
    " 2>/dev/null || echo '[]')

    # Get row counts
    local COUNTS=$(docker exec -e PGPASSWORD=plane $DB_CONTAINER psql -U plane -d plane -t -c "
        SELECT json_object_agg(tablename, n_live_tup)
        FROM pg_stat_user_tables;
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"plane\",
        \"database_type\": \"postgresql\",
        \"database_name\": \"plane\",
        \"url\": \"http://localhost:80\",
        \"tables\": $SCHEMA,
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Plane schema saved to $OUTPUT_FILE"
}

# ============================================================================
# ZAMMAD - Helpdesk (PostgreSQL)
# ============================================================================
fetch_zammad_schema() {
    log_info "Fetching Zammad PostgreSQL schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/zammad_schema.json"
    local DB_CONTAINER=$(docker ps --format "{{.Names}}" | grep "zammad.*postgres" | head -1)

    if [ -z "$DB_CONTAINER" ]; then
        log_warn "Zammad database container not found"
        echo '{"error": "Container not running", "service": "zammad"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Get all tables
    local TABLES=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(tablename)
        FROM pg_tables
        WHERE schemaname = 'public';
    " 2>/dev/null || echo '[]')

    # Get detailed schema grouped by table
    local SCHEMA=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(
            json_build_object(
                'name', t.table_name,
                'columns', (
                    SELECT json_agg(
                        json_build_object(
                            'column_name', c.column_name,
                            'ordinal_position', c.ordinal_position,
                            'column_default', c.column_default,
                            'is_nullable', c.is_nullable,
                            'data_type', c.data_type,
                            'character_maximum_length', c.character_maximum_length,
                            'character_octet_length', c.character_octet_length,
                            'numeric_precision', c.numeric_precision,
                            'numeric_precision_radix', c.numeric_precision_radix,
                            'numeric_scale', c.numeric_scale,
                            'datetime_precision', c.datetime_precision,
                            'character_set_name', c.character_set_name,
                            'collation_name', c.collation_name,
                            'udt_name', c.udt_name,
                            'is_updatable', c.is_updatable
                        )
                        ORDER BY c.ordinal_position
                    )
                    FROM information_schema.columns c
                    WHERE c.table_schema = 'public'
                    AND c.table_name = t.table_name
                )
            )
        )
        FROM (SELECT DISTINCT table_name FROM information_schema.columns WHERE table_schema = 'public') t;
    " 2>/dev/null || echo '[]')

    # Get row counts
    local COUNTS=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_object_agg(tablename, n_live_tup)
        FROM pg_stat_user_tables;
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"zammad\",
        \"database_type\": \"postgresql\",
        \"database_name\": \"zammad_production\",
        \"url\": \"http://localhost:8083\",
        \"tables\": $SCHEMA,
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Zammad schema saved to $OUTPUT_FILE"
}

# ============================================================================
# DOLIBARR - CRM/ERP (MariaDB)
# ============================================================================
fetch_dolibarr_schema() {
    log_info "Fetching Dolibarr MariaDB schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/dolibarr_schema.json"
    local DB_CONTAINER=$(docker ps --format "{{.Names}}" | grep "dolibarr.*maria" | head -1)

    if [ -z "$DB_CONTAINER" ]; then
        log_warn "Dolibarr database container not found"
        echo '{"error": "Container not running", "service": "dolibarr"}' > "$OUTPUT_FILE"
        return 1
    fi

    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    local DB_NAME="dolidb"
    local DB_USER="root"
    local DB_PASS="root"

    # Get all tables
    local TABLES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT JSON_ARRAYAGG(TABLE_NAME)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = '$DB_NAME';
    " 2>/dev/null || echo '[]')

    # Get detailed schema grouped by table
    local SCHEMA=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', TABLE_NAME,
                'columns', (
                    SELECT JSON_ARRAYAGG(
                        JSON_OBJECT(
                            'column_name', COLUMN_NAME,
                            'ordinal_position', ORDINAL_POSITION,
                            'column_default', COLUMN_DEFAULT,
                            'is_nullable', IS_NULLABLE,
                            'data_type', DATA_TYPE,
                            'character_maximum_length', CHARACTER_MAXIMUM_LENGTH,
                            'character_octet_length', CHARACTER_OCTET_LENGTH,
                            'numeric_precision', NUMERIC_PRECISION,
                            'numeric_scale', NUMERIC_SCALE,
                            'datetime_precision', DATETIME_PRECISION,
                            'character_set_name', CHARACTER_SET_NAME,
                            'collation_name', COLLATION_NAME,
                            'column_type', COLUMN_TYPE,
                            'column_key', COLUMN_KEY,
                            'extra', EXTRA,
                            'privileges', PRIVILEGES,
                            'column_comment', COLUMN_COMMENT,
                            'generation_expression', GENERATION_EXPRESSION
                        )
                        ORDER BY ORDINAL_POSITION
                    )
                    FROM information_schema.COLUMNS c2
                    WHERE c2.TABLE_SCHEMA = '$DB_NAME'
                    AND c2.TABLE_NAME = t.TABLE_NAME
                )
            )
        )
        FROM (SELECT DISTINCT TABLE_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = '$DB_NAME') t;
    " 2>/dev/null || echo '[]')

    # Get row counts
    local COUNTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT JSON_OBJECTAGG(TABLE_NAME, TABLE_ROWS)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = '$DB_NAME';
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"dolibarr\",
        \"database_type\": \"mariadb\",
        \"database_name\": \"$DB_NAME\",
        \"url\": \"http://localhost:8082\",
        \"tables\": $SCHEMA,
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Dolibarr schema saved to $OUTPUT_FILE"
}

# ============================================================================
# FRAPPE - ERP system (MariaDB)
# ============================================================================
fetch_frappe_schema() {
    log_info "Fetching Frappe MariaDB schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/frappe_schema.json"
    local DB_CONTAINER="docker-mariadb-1"
    local DB_USER="root"
    local DB_PASS="123"

    if ! docker ps | grep -q "$DB_CONTAINER"; then
        log_warn "Frappe database container not found"
        echo '{"error": "Container not running", "service": "frappe"}' > "$OUTPUT_FILE"
        return 1
    fi

    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    # Get Frappe site databases
    local FRAPPE_DBS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS -e "SHOW DATABASES;" 2>/dev/null | grep "^_")

    if [ -z "$FRAPPE_DBS" ]; then
        log_warn "No Frappe site databases found"
        echo '{"error": "No databases found", "service": "frappe"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Find database with most data
    local MAX_COUNT=0
    local SITE_DB=""
    for db in $FRAPPE_DBS; do
        local COUNT=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $db -se "
            SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA='$db';
        " 2>/dev/null || echo "0")
        if [ "$COUNT" -gt "$MAX_COUNT" ]; then
            MAX_COUNT=$COUNT
            SITE_DB=$db
        fi
    done

    log_info "Using Frappe database: $SITE_DB"

    # Get all tables
    local TABLES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT JSON_ARRAYAGG(TABLE_NAME)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = '$SITE_DB'
        AND TABLE_NAME LIKE 'tab%';
    " 2>/dev/null || echo '[]')

    # Get detailed schema grouped by table (all tab* tables)
    local SCHEMA=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', TABLE_NAME,
                'columns', (
                    SELECT JSON_ARRAYAGG(
                        JSON_OBJECT(
                            'column_name', COLUMN_NAME,
                            'ordinal_position', ORDINAL_POSITION,
                            'column_default', COLUMN_DEFAULT,
                            'is_nullable', IS_NULLABLE,
                            'data_type', DATA_TYPE,
                            'character_maximum_length', CHARACTER_MAXIMUM_LENGTH,
                            'character_octet_length', CHARACTER_OCTET_LENGTH,
                            'numeric_precision', NUMERIC_PRECISION,
                            'numeric_scale', NUMERIC_SCALE,
                            'datetime_precision', DATETIME_PRECISION,
                            'character_set_name', CHARACTER_SET_NAME,
                            'collation_name', COLLATION_NAME,
                            'column_type', COLUMN_TYPE,
                            'column_key', COLUMN_KEY,
                            'extra', EXTRA,
                            'privileges', PRIVILEGES,
                            'column_comment', COLUMN_COMMENT,
                            'generation_expression', GENERATION_EXPRESSION
                        )
                        ORDER BY ORDINAL_POSITION
                    )
                    FROM information_schema.COLUMNS c2
                    WHERE c2.TABLE_SCHEMA = '$SITE_DB'
                    AND c2.TABLE_NAME = t.TABLE_NAME
                )
            )
        )
        FROM (SELECT DISTINCT TABLE_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = '$SITE_DB' AND TABLE_NAME LIKE 'tab%') t;
    " 2>/dev/null || echo '[]')

    # Get row counts for DocType tables
    local COUNTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT JSON_OBJECTAGG(TABLE_NAME, TABLE_ROWS)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = '$SITE_DB'
        AND TABLE_NAME LIKE 'tab%';
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"frappe_hrms\",
        \"database_type\": \"mariadb\",
        \"database_name\": \"$SITE_DB\",
        \"url\": \"http://localhost:8084\",
        \"tables\": $SCHEMA,
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Frappe schema saved to $OUTPUT_FILE"
}

# ============================================================================
# GITLAB - DevOps platform (PostgreSQL)
# ============================================================================
fetch_gitlab_schema() {
    log_info "Fetching GitLab PostgreSQL schema..."

    local OUTPUT_FILE="$OUTPUT_DIR/gitlab_schema.json"

    # Check if GitLab container is running
    if ! docker ps --format "{{.Names}}" | grep -q "^gitlab$"; then
        log_warn "GitLab container is not running"
        echo '{"error": "Container not running", "service": "gitlab"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Use Python to properly group schema by table (same format as other apps)
    python3 << 'PYEOF' > "$OUTPUT_FILE"
import subprocess
import json
from datetime import datetime
import sys

def run_sql(sql):
    """Execute SQL in GitLab PostgreSQL"""
    cmd = ['docker', 'exec', 'gitlab', 'gitlab-psql', '-d', 'gitlabhq_production', '-t', '-A', '-c', sql]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return json.loads(output)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

# Get all table names
table_names = run_sql("""
    SELECT json_agg(tablename ORDER BY tablename)
    FROM pg_tables
    WHERE schemaname = 'public';
""")

if not table_names:
    print(json.dumps({"error": "Could not fetch tables"}))
    exit(1)

print(f"Fetching schema for {len(table_names)} tables...", file=sys.stderr)

# Build schema in same format as other apps: list of {name, columns}
tables = []
for i, table_name in enumerate(table_names):
    if (i + 1) % 100 == 0:
        print(f"  Processing {i+1}/{len(table_names)}...", file=sys.stderr)

    # Get columns for this table
    columns = run_sql(f"""
        SELECT COALESCE(json_agg(row_to_json(t) ORDER BY ordinal_position), '[]'::json)
        FROM (
            SELECT
                column_name,
                ordinal_position,
                column_default,
                is_nullable,
                data_type,
                character_maximum_length,
                character_octet_length,
                numeric_precision,
                numeric_precision_radix,
                numeric_scale,
                datetime_precision,
                character_set_name,
                collation_name,
                udt_name,
                is_updatable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table_name}'
        ) t;
    """)

    tables.append({
        "name": table_name,
        "columns": columns if columns else []
    })

# Build final schema (same format as other apps)
schema = {
    "service": "gitlab",
    "database_type": "postgresql",
    "database_name": "gitlabhq_production",
    "url": "http://localhost:8080",
    "tables": tables,
    "timestamp": datetime.now().isoformat()
}

print(json.dumps(schema, indent=2))
PYEOF

    log_info "GitLab schema saved to $OUTPUT_FILE"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting database schema extraction..."
    log_info "Output directory: $OUTPUT_DIR"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Fetch schemas from all services
    fetch_rocketchat_schema || log_warn "RocketChat schema fetch failed"
    echo ""

    fetch_owncloud_schema || log_warn "OwnCloud schema fetch failed"
    echo ""

    fetch_plane_schema || log_warn "Plane schema fetch failed"
    echo ""

    fetch_zammad_schema || log_warn "Zammad schema fetch failed"
    echo ""

    fetch_dolibarr_schema || log_warn "Dolibarr schema fetch failed"
    echo ""

    fetch_frappe_schema || log_warn "Frappe schema fetch failed"
    echo ""

    fetch_gitlab_schema || log_warn "GitLab schema fetch failed"
    echo ""

    log_info "✓ Database schema extraction complete!"
    log_info "Schema files available in: $OUTPUT_DIR"
    echo ""
    log_info "Files created:"
    ls -lh "$OUTPUT_DIR"
}

# Show usage if --help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    cat << EOF
Usage: $0

Extract database schemas (tables and columns) from all applications.

This script extracts:
- List of all tables/collections
- Column names and data types
- Constraints (primary keys, nullable, defaults)
- Row counts per table

OUTPUT:
    Saves JSON files to: $OUTPUT_DIR/
    One file per application: {service}_schema.json

EXAMPLES:
    # Extract all schemas
    ./fetch_database_schema.sh

    # View OwnCloud schema
    cat $OUTPUT_DIR/owncloud_schema.json | jq .

    # List all Plane tables
    cat $OUTPUT_DIR/plane_schema.json | jq '.tables[]'

    # See column details for a specific table
    cat $OUTPUT_DIR/dolibarr_schema.json | jq '.schema[] | select(.table_name == "llx_user")'

EOF
    exit 0
fi

# Run main function
main "$@"
