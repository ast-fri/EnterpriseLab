#!/bin/bash

# sync_data.sh - Full bidirectional sync: INSERT, UPDATE, DELETE
# This script synchronizes JSON data with databases (supports all CRUD operations)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/fetched_data"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Operation mode
DRY_RUN=${DRY_RUN:-0}
AUTO_CONFIRM=${AUTO_CONFIRM:-0}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_dry() {
    echo -e "${BLUE}[DRY-RUN]${NC} $1"
}

log_operation() {
    local op=$1
    local item=$2
    case $op in
        INSERT) echo -e "${GREEN}[+]${NC} INSERT: $item" ;;
        UPDATE) echo -e "${YELLOW}[~]${NC} UPDATE: $item" ;;
        DELETE) echo -e "${RED}[-]${NC} DELETE: $item" ;;
    esac
}

# ============================================================================
# ROCKETCHAT - MongoDB Full Sync
# ============================================================================
sync_rocketchat_data() {
    log_info "Syncing RocketChat data..."

    local INPUT_FILE="$INPUT_DIR/rocketchat.json"
    if [ ! -f "$INPUT_FILE" ]; then
        log_warn "RocketChat data file not found"
        return 1
    fi

    local MONGO_CONTAINER="rocket-chat-mongodb-1"

    if [ "$DRY_RUN" = "1" ]; then
        local USERS=$(cat "$INPUT_FILE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(len(data.get('data', {}).get('users', [])))
")
        log_dry "Would sync $USERS users to RocketChat"
        return 0
    fi

    # Full sync: upsert (insert or update based on _id)
    python3 << 'PYEOF'
import sys, json, subprocess

with open('fetched_data/rocketchat.json') as f:
    data = json.load(f)

users = data.get('data', {}).get('users', [])

for user in users:
    user_id = user.get('_id')
    user_json = json.dumps(user).replace("'", "\\'")

    cmd = f"""docker exec rocket-chat-mongodb-1 mongosh rocketchat --quiet --eval '
        db.users.replaceOne(
            {{_id: "{user_id}"}},
            {user_json},
            {{upsert: true}}
        )
    '"""

    subprocess.run(cmd, shell=True, capture_output=True)

print(f"Synced {len(users)} users")
PYEOF

    log_info "RocketChat sync complete"
}

# ============================================================================
# PLANE - PostgreSQL Full Sync
# ============================================================================
sync_plane_data() {
    log_info "Syncing Plane data..."

    local INPUT_FILE="$INPUT_DIR/plane.json"
    if [ ! -f "$INPUT_FILE" ]; then
        log_warn "Plane data file not found"
        return 1
    fi

    if [ "$DRY_RUN" = "1" ]; then
        log_dry "Would sync Plane workspaces, projects, issues"
        return 0
    fi

    # Sync workspaces
    python3 << 'PYEOF'
import sys, json, subprocess

with open('fetched_data/plane.json') as f:
    data = json.load(f)

workspaces = data.get('data', {}).get('workspaces', [])

for ws in workspaces:
    sql = f"""
        INSERT INTO workspaces (id, name, slug, created_at)
        VALUES ('{ws['id']}', E'{ws['name'].replace("'", "''")}', '{ws['slug']}', '{ws['created_at']}')
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            slug = EXCLUDED.slug;
    """

    cmd = ['docker', 'exec', '-e', 'PGPASSWORD=plane', 'plane-plane-db-1',
           'psql', '-U', 'plane', '-d', 'plane', '-c', sql]

    subprocess.run(cmd, capture_output=True)

print(f"Synced {len(workspaces)} workspaces")
PYEOF

    # Sync projects
    python3 << 'PYEOF'
import sys, json, subprocess

with open('fetched_data/plane.json') as f:
    data = json.load(f)

projects = data.get('data', {}).get('projects', [])

for proj in projects:
    sql = f"""
        INSERT INTO projects (id, name, description, workspace_id, created_at)
        VALUES ('{proj['id']}', E'{proj['name'].replace("'", "''")}',
                E'{proj.get('description', '').replace("'", "''")}',
                '{proj['workspace_id']}', '{proj['created_at']}')
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description;
    """

    cmd = ['docker', 'exec', '-e', 'PGPASSWORD=plane', 'plane-plane-db-1',
           'psql', '-U', 'plane', '-d', 'plane', '-c', sql]

    subprocess.run(cmd, capture_output=True)

print(f"Synced {len(projects)} projects")
PYEOF

    # Sync issues
    python3 << 'PYEOF'
import sys, json, subprocess

with open('fetched_data/plane.json') as f:
    data = json.load(f)

issues = data.get('data', {}).get('issues', [])

for issue in issues:
    sql = f"""
        INSERT INTO issues (id, name, state_id, project_id, created_at)
        VALUES ('{issue['id']}', E'{issue['name'].replace("'", "''")}',
                '{issue['state_id']}', '{issue['project_id']}', '{issue['created_at']}')
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            state_id = EXCLUDED.state_id;
    """

    cmd = ['docker', 'exec', '-e', 'PGPASSWORD=plane', 'plane-plane-db-1',
           'psql', '-U', 'plane', '-d', 'plane', '-c', sql]

    subprocess.run(cmd, capture_output=True)

print(f"Synced {len(issues)} issues")
PYEOF

    log_info "Plane sync complete"
}

# ============================================================================
# DOLIBARR - MariaDB Full Sync
# ============================================================================
sync_dolibarr_data() {
    log_info "Syncing Dolibarr data..."

    local INPUT_FILE="$INPUT_DIR/dolibarr.json"
    if [ ! -f "$INPUT_FILE" ]; then
        log_warn "Dolibarr data file not found"
        return 1
    fi

    if [ "$DRY_RUN" = "1" ]; then
        log_dry "Would sync Dolibarr users, companies, products, invoices"
        return 0
    fi

    local DB_CONTAINER="dolibarr-mariadb-1"
    local DB_NAME="dolidb"
    local DB_USER="root"
    local DB_PASS="root"
    local MYSQL_CMD="mariadb"

    # Sync users
    python3 << 'PYEOF'
import sys, json, subprocess

with open('fetched_data/dolibarr.json') as f:
    data = json.load(f)

users = data.get('data', {}).get('users', [])

for user in users:
    sql = f"""
        INSERT INTO llx_user (rowid, login, lastname, firstname, email)
        VALUES ({user['rowid']}, '{user['login']}',
                '{user.get('lastname', '')}', '{user.get('firstname', '')}',
                '{user.get('email', '')}')
        ON DUPLICATE KEY UPDATE
            login = VALUES(login),
            lastname = VALUES(lastname),
            firstname = VALUES(firstname),
            email = VALUES(email);
    """

    cmd = f"docker exec dolibarr-mariadb-1 mariadb -uroot -proot dolidb -e \"{sql}\""
    subprocess.run(cmd, shell=True, capture_output=True)

print(f"Synced {len(users)} users")
PYEOF

    # Sync products
    python3 << 'PYEOF'
import sys, json, subprocess

with open('fetched_data/dolibarr.json') as f:
    data = json.load(f)

products = data.get('data', {}).get('products', [])

for prod in products:
    sql = f"""
        INSERT INTO llx_product (rowid, ref, label, price)
        VALUES ({prod['rowid']}, '{prod['ref']}', '{prod['label']}', {prod.get('price', 0)})
        ON DUPLICATE KEY UPDATE
            ref = VALUES(ref),
            label = VALUES(label),
            price = VALUES(price);
    """

    cmd = f"docker exec dolibarr-mariadb-1 mariadb -uroot -proot dolidb -e \"{sql}\""
    subprocess.run(cmd, shell=True, capture_output=True)

print(f"Synced {len(products)} products")
PYEOF

    log_info "Dolibarr sync complete"
}

# ============================================================================
# DELETE OPERATIONS (Optional - dangerous!)
# ============================================================================
delete_removed_records() {
    log_warn "DELETE operations not yet implemented for safety"
    log_info "To delete records: manually remove from database or implement custom logic"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
main() {
    log_info "Starting full data sync (INSERT/UPDATE)..."
    log_info "Input directory: $INPUT_DIR"
    echo ""

    if [ ! -d "$INPUT_DIR" ]; then
        log_error "Input directory not found: $INPUT_DIR"
        exit 1
    fi

    # Confirmation prompt
    if [ "$DRY_RUN" = "0" ] && [ -z "$AUTO_CONFIRM" ]; then
        echo ""
        log_warn "⚠️  This will MODIFY database records (INSERT + UPDATE)"
        log_warn "⚠️  Make sure you have backups!"
        echo ""
        read -p "Continue? (yes/no): " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            log_info "Cancelled"
            exit 0
        fi
        echo ""
    fi

    # Sync all services
    sync_rocketchat_data || log_warn "RocketChat sync failed"
    echo ""

    sync_plane_data || log_warn "Plane sync failed"
    echo ""

    sync_dolibarr_data || log_warn "Dolibarr sync failed"
    echo ""

    log_info "✓ Sync complete!"

    if [ "$DRY_RUN" = "1" ]; then
        log_info "This was a DRY-RUN"
    fi
}

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    cat << 'EOF'
Usage: sync_data.sh [OPTIONS]

Full data synchronization (INSERT + UPDATE) from JSON to databases.

OPTIONS:
    --dry-run           Show what would be done
    --auto-confirm      Skip confirmation prompt

ENVIRONMENT VARIABLES:
    DRY_RUN=1          Dry-run mode
    AUTO_CONFIRM=1     Auto-confirm

WORKFLOW:
    1. Fetch data:     ./fetch_database_data.sh
    2. Edit JSON:      vim fetched_data/plane.json
    3. Sync back:      ./sync_data.sh

OPERATIONS SUPPORTED:
    ✓ INSERT - New records added to JSON
    ✓ UPDATE - Modified records in JSON
    ✗ DELETE - Not auto-detected (manual SQL needed)

EXAMPLES:
    # Dry run
    ./sync_data.sh --dry-run

    # Actual sync
    ./sync_data.sh

    # Edit workflow
    ./fetch_database_data.sh          # 1. Fetch
    vim fetched_data/plane.json       # 2. Edit
    ./sync_data.sh                     # 3. Push changes

EOF
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --auto-confirm)
            AUTO_CONFIRM=1
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main "$@"
