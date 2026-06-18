#!/bin/bash

# fetch_database_data.sh - Fetch actual application data from databases
# This script directly queries application databases to fetch real data
# (users, projects, messages, tickets, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/fetched_data"

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

# Helper function to format JSON using Python
format_json() {
    local input=$(cat)
    if [ -z "$input" ]; then
        echo "{}"
        return
    fi
    echo "$input" | python3 -m json.tool 2>/dev/null || echo "$input"
}

# ============================================================================
# ROCKETCHAT - Team communication platform (MongoDB)
# ============================================================================
fetch_rocketchat_data() {
    log_info "Fetching RocketChat data from MongoDB..."

    local OUTPUT_FILE="$OUTPUT_DIR/rocketchat.json"

    # Use the correct MongoDB container with actual data
    local MONGO_CONTAINER="rocket-chat-mongodb-1"

    # Get users
    local USERS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.users.find({}, {
            username: 1,
            name: 1,
            emails: 1,
            roles: 1,
            status: 1,
            active: 1
        }).toArray())
    ' 2>/dev/null || echo '[]')

    # Get rooms/channels
    local ROOMS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.rocketchat_room.find({}, {
            name: 1,
            t: 1,
            msgs: 1,
            usersCount: 1,
            ts: 1,
            _updatedAt: 1
        }).toArray())
    ' 2>/dev/null || echo '[]')

    # Get recent messages
    local MESSAGES=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.rocketchat_message.find({}, {
            msg: 1,
            rid: 1,
            u: 1,
            ts: 1
        }).toArray())
    ' 2>/dev/null || echo '[]')

    # Get subscriptions
    local SUBSCRIPTIONS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.rocketchat_subscription.find({}, {
            rid: 1,
            u: 1,
            name: 1,
            t: 1,
            alert: 1,
            unread: 1
        }).toArray())
    ' 2>/dev/null || echo '[]')

    # Get roles
    local ROLES=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.rocketchat_roles.find({}, {
            name: 1,
            description: 1,
            scope: 1,
            mandatory2fa: 1
        }).toArray())
    ' 2>/dev/null || echo '[]')

    # Get integrations
    local INTEGRATIONS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify(db.rocketchat_integration.find({}, {
            name: 1,
            type: 1,
            enabled: 1,
            channel: 1,
            event: 1
        }).toArray())
    ' 2>/dev/null || echo '[]')

    # Get statistics
    local STATS=$(docker exec $MONGO_CONTAINER mongosh rocketchat --quiet --eval '
        JSON.stringify({
            totalUsers: db.users.countDocuments(),
            totalRooms: db.rocketchat_room.countDocuments(),
            totalMessages: db.rocketchat_message.countDocuments(),
            totalSubscriptions: db.rocketchat_subscription.countDocuments(),
            totalRoles: db.rocketchat_roles.countDocuments(),
            totalIntegrations: db.rocketchat_integration.countDocuments()
        })
    ' 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"rocketchat\",
        \"url\": \"http://localhost:3000\",
        \"database\": \"mongodb\",
        \"data\": {
            \"statistics\": $STATS,
            \"users\": $USERS,
            \"rooms\": $ROOMS,
            \"recent_messages\": $MESSAGES,
            \"subscriptions\": $SUBSCRIPTIONS,
            \"roles\": $ROLES,
            \"integrations\": $INTEGRATIONS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "RocketChat data saved to $OUTPUT_FILE"
}

# ============================================================================
# OWNCLOUD - File sharing (MariaDB)
# ============================================================================
fetch_owncloud_data() {
    log_info "Fetching OwnCloud data from MariaDB..."

    local OUTPUT_FILE="$OUTPUT_DIR/owncloud.json"

    # Correct container name
    local DB_CONTAINER="owncloud-db-1"

    if ! docker ps | grep -q "$DB_CONTAINER"; then
        log_warn "OwnCloud database container not found"
        echo '{"error": "Database not accessible", "service": "owncloud"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Try both mysql and mariadb commands
    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    # Get users
    local USERS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'uid', uid,
                'displayname', displayname,
                'password', IF(password IS NOT NULL, 'SET', 'NOT SET')
            )
        ) FROM oc_users ;
    " 2>/dev/null || echo '[]')

    # Get file storage statistics
    local STORAGE=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT JSON_OBJECT(
            'total_files', COUNT(*),
            'total_size', SUM(size)
        ) FROM oc_filecache;
    " 2>/dev/null || echo '{}')

    # Get shares
    local SHARES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'id', id,
                'share_type', share_type,
                'share_with', share_with,
                'uid_owner', uid_owner,
                'file_target', file_target
            )
        ), JSON_ARRAY()) FROM oc_share ;
    " 2>/dev/null || echo '[]')

    # Get groups
    local OWNCLOUD_GROUPS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'gid', gid
            )
        ), JSON_ARRAY()) FROM oc_groups;
    " 2>/dev/null || echo '[]')

    # Get group memberships
    local GROUP_USERS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'gid', gid,
                'uid', uid
            )
        ), JSON_ARRAY()) FROM oc_group_user ;
    " 2>/dev/null || echo '[]')

    # Get recent activity
    local ACTIVITY=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'activity_id', activity_id,
                'timestamp', timestamp,
                'subject', subject,
                'user', \`user\`,
                'type', type
            )
        ), JSON_ARRAY()) FROM oc_activity ;
    " 2>/dev/null || echo '[]')

    # Get comments
    local COMMENTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u owncloud -psecret owncloud -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'id', id,
                'parent_id', parent_id,
                'actor_id', actor_id,
                'message', message,
                'creation_timestamp', creation_timestamp
            )
        ), JSON_ARRAY()) FROM oc_comments ;
    " 2>/dev/null || echo '[]')

    echo "{
        \"service\": \"owncloud\",
        \"url\": \"http://localhost:8081\",
        \"database\": \"mariadb\",
        \"data\": {
            \"users\": $USERS,
            \"storage_stats\": $STORAGE,
            \"shares\": $SHARES,
            \"groups\": $OWNCLOUD_GROUPS,
            \"group_users\": $GROUP_USERS,
            \"recent_activity\": $ACTIVITY,
            \"comments\": $COMMENTS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "OwnCloud data saved to $OUTPUT_FILE"
}

# ============================================================================
# GITLAB - DevOps platform (PostgreSQL)
# ============================================================================
fetch_gitlab_data() {
    log_info "Fetching GitLab data..."

    local OUTPUT_FILE="$OUTPUT_DIR/gitlab.json"

    # Check if GitLab container is running
    local GITLAB_STATUS=$(docker ps -a --format "{{.Names}}\t{{.Status}}" | grep "^gitlab\s" | awk '{print $2}')

    if [ "$GITLAB_STATUS" != "Up" ]; then
        log_warn "GitLab container is not running (Status: $GITLAB_STATUS)"
        echo "{
            \"service\": \"gitlab\",
            \"url\": \"http://localhost:8080\",
            \"error\": \"Container not running\",
            \"status\": \"$GITLAB_STATUS\",
            \"note\": \"GitLab uses internal PostgreSQL database. Container needs to be running to access data.\",
            \"timestamp\": \"$(date -Iseconds)\"
        }" | format_json > "$OUTPUT_FILE"
        return 1
    fi

    # If running, try to fetch data via API
    local PROJECTS=$(curl -s "http://localhost:8080/api/v4/projects?simple=true&per_page=10" 2>/dev/null || echo '[]')

    echo "{
        \"service\": \"gitlab\",
        \"url\": \"http://localhost:8080\",
        \"database\": \"postgresql (internal)\",
        \"note\": \"GitLab uses internal PostgreSQL. API access requires authentication token.\",
        \"data\": {
            \"projects\": $PROJECTS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "GitLab data saved to $OUTPUT_FILE"
}

# ============================================================================
# PLANE - Project management (PostgreSQL)
# ============================================================================
fetch_plane_data() {
    log_info "Fetching Plane data from PostgreSQL..."

    local OUTPUT_FILE="$OUTPUT_DIR/plane.json"

    if ! docker ps | grep -q "plane.*db\|plane.*postgres"; then
        log_warn "Plane database container not found"
        echo '{"error": "Database not accessible", "service": "plane"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Get workspaces
    local WORKSPACES=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, name, slug, created_at
            FROM workspaces
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get projects
    local PROJECTS=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, name, description, workspace_id, created_at
            FROM projects
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get issues
    local ISSUES=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, name, state_id, project_id, created_at
            FROM issues
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get cycles
    local CYCLES=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, name, project_id, start_date, end_date, created_at
            FROM cycles
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get modules
    local MODULES=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, name, description, project_id, start_date, target_date, created_at
            FROM modules
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get issue assignees
    local ASSIGNEES=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, issue_id, user_id, created_at
            FROM issue_assignees
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get accounts
    local ACCOUNTS=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json) FROM (
            SELECT id, provider, provider_account_id, user_id, created_at
            FROM accounts
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get statistics
    local STATS=$(docker exec -e PGPASSWORD=plane plane-plane-db-1 psql -U plane -d plane -t -c "
        SELECT row_to_json(stats) FROM (
            SELECT
                (SELECT COUNT(*) FROM workspaces) as total_workspaces,
                (SELECT COUNT(*) FROM projects) as total_projects,
                (SELECT COUNT(*) FROM issues) as total_issues,
                (SELECT COUNT(*) FROM cycles) as total_cycles,
                (SELECT COUNT(*) FROM modules) as total_modules,
                (SELECT COUNT(*) FROM issue_assignees) as total_assignees
        ) stats;
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"plane\",
        \"url\": \"http://localhost:80\",
        \"database\": \"postgresql\",
        \"data\": {
            \"statistics\": $STATS,
            \"workspaces\": $WORKSPACES,
            \"projects\": $PROJECTS,
            \"issues\": $ISSUES,
            \"cycles\": $CYCLES,
            \"modules\": $MODULES,
            \"assignees\": $ASSIGNEES,
            \"accounts\": $ACCOUNTS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Plane data saved to $OUTPUT_FILE"
}

# ============================================================================
# ZAMMAD - Helpdesk (PostgreSQL)
# ============================================================================
fetch_zammad_data() {
    log_info "Fetching Zammad data from PostgreSQL..."

    local OUTPUT_FILE="$OUTPUT_DIR/zammad.json"

    if ! docker ps | grep -q "zammad.*postgres"; then
        log_warn "Zammad database container not found"
        echo '{"error": "Database not accessible", "service": "zammad"}' > "$OUTPUT_FILE"
        return 1
    fi

    local DB_CONTAINER=$(docker ps --format "{{.Names}}" | grep "zammad.*postgres" | head -1)

    # Get tickets
    local TICKETS=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, title, state_id, priority_id, group_id, created_at
            FROM tickets
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get users
    local USERS=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, login, firstname, lastname, email, active
            FROM users
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get organizations
    local ORGS=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, name, active, created_at
            FROM organizations
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get ticket articles (messages/replies)
    local ARTICLES=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, ticket_id, from_field as sender, subject, body, internal, created_at
            FROM ticket_articles
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get groups
    local ZAMMAD_GROUPS=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, name, active, created_at
            FROM groups
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get roles
    local ROLES=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, name, active, created_at
            FROM roles
            
        ) t;
    " 2>/dev/null || echo '[]')

    # Get tags
    local TAGS=$(docker exec $DB_CONTAINER psql -U zammad -d zammad_production -t -c "
        SELECT json_agg(row_to_json(t)) FROM (
            SELECT id, name, name_downcase, created_at
            FROM tags
            
        ) t;
    " 2>/dev/null || echo '[]')

    echo "{
        \"service\": \"zammad\",
        \"url\": \"http://localhost:8083\",
        \"database\": \"postgresql\",
        \"data\": {
            \"tickets\": $TICKETS,
            \"users\": $USERS,
            \"organizations\": $ORGS,
            \"ticket_articles\": $ARTICLES,
            \"groups\": $ZAMMAD_GROUPS,
            \"roles\": $ROLES,
            \"tags\": $TAGS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Zammad data saved to $OUTPUT_FILE"
}

# ============================================================================
# DOLIBARR - CRM/ERP (MariaDB)
# ============================================================================
fetch_dolibarr_data() {
    log_info "Fetching Dolibarr data from MariaDB..."

    local OUTPUT_FILE="$OUTPUT_DIR/dolibarr.json"

    if ! docker ps | grep -q "dolibarr.*maria"; then
        log_warn "Dolibarr database container not found"
        echo '{"error": "Database not accessible", "service": "dolibarr"}' > "$OUTPUT_FILE"
        return 1
    fi

    local DB_CONTAINER=$(docker ps --format "{{.Names}}" | grep "dolibarr.*maria" | head -1)

    # Try both mysql and mariadb commands
    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    # Correct database name is 'dolidb' not 'dolibarr'
    local DB_NAME="dolidb"
    # Use root credentials
    local DB_USER="root"
    local DB_PASS="root"

    # Get users
    local USERS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'login', login,
                'lastname', lastname,
                'firstname', firstname,
                'email', email
            )
        ), JSON_ARRAY()) FROM llx_user ;
    " 2>/dev/null || echo '[]')

    # Get products
    local PRODUCTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'ref', ref,
                'label', label,
                'price', price
            )
        ), JSON_ARRAY()) FROM llx_product ;
    " 2>/dev/null || echo '[]')

    # Get companies (societe)
    local COMPANIES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'nom', nom,
                'email', email,
                'client', client,
                'fournisseur', fournisseur
            )
        ), JSON_ARRAY()) FROM llx_societe ;
    " 2>/dev/null || echo '[]')

    # Get contacts (socpeople)
    local CONTACTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'lastname', lastname,
                'firstname', firstname,
                'email', email,
                'fk_soc', fk_soc
            )
        ), JSON_ARRAY()) FROM llx_socpeople ;
    " 2>/dev/null || echo '[]')

    # Get invoices
    local INVOICES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'ref', ref,
                'fk_soc', fk_soc,
                'total_ht', total_ht,
                'total_ttc', total_ttc,
                'paye', paye
            )
        ), JSON_ARRAY()) FROM llx_facture ;
    " 2>/dev/null || echo '[]')

    # Get orders
    local ORDERS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'ref', ref,
                'fk_soc', fk_soc,
                'total_ht', total_ht,
                'total_ttc', total_ttc
            )
        ), JSON_ARRAY()) FROM llx_commande ;
    " 2>/dev/null || echo '[]')

    # Get proposals
    local PROPOSALS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'ref', ref,
                'fk_soc', fk_soc,
                'total_ht', total_ht,
                'total_ttc', total_ttc
            )
        ), JSON_ARRAY()) FROM llx_propal ;
    " 2>/dev/null || echo '[]')

    # Get projects
    local PROJECTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $DB_NAME -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'rowid', rowid,
                'ref', ref,
                'title', title,
                'fk_soc', fk_soc,
                'fk_statut', fk_statut
            )
        ), JSON_ARRAY()) FROM llx_projet ;
    " 2>/dev/null || echo '[]')

    echo "{
        \"service\": \"dolibarr\",
        \"url\": \"http://localhost:8082\",
        \"database\": \"mariadb\",
        \"data\": {
            \"users\": $USERS,
            \"products\": $PRODUCTS,
            \"companies\": $COMPANIES,
            \"contacts\": $CONTACTS,
            \"invoices\": $INVOICES,
            \"orders\": $ORDERS,
            \"proposals\": $PROPOSALS,
            \"projects\": $PROJECTS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Dolibarr data saved to $OUTPUT_FILE"
}

# ============================================================================
# FRAPPE - ERP system (MariaDB)
# ============================================================================
fetch_frappe_data() {
    log_info "Fetching Frappe data from MariaDB..."

    local OUTPUT_FILE="$OUTPUT_DIR/frappe.json"

    # Frappe uses docker-mariadb-1 with multiple site databases
    local DB_CONTAINER="docker-mariadb-1"
    local DB_USER="root"
    local DB_PASS="123"

    if ! docker ps | grep -q "$DB_CONTAINER"; then
        log_warn "Frappe database container not found"
        echo '{"error": "Database not accessible", "service": "frappe"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Try both mysql and mariadb commands
    local MYSQL_CMD="mysql"
    docker exec $DB_CONTAINER which mariadb &>/dev/null && MYSQL_CMD="mariadb"

    # Get list of Frappe site databases (they have hash names starting with _)
    local FRAPPE_DBS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS -e "SHOW DATABASES;" 2>/dev/null | grep "^_")

    if [ -z "$FRAPPE_DBS" ]; then
        log_warn "No Frappe site databases found"
        echo '{"error": "No Frappe databases found", "service": "frappe"}' > "$OUTPUT_FILE"
        return 1
    fi

    # Find the database with the most data
    local SITE_DB=""
    local MAX_COUNT=0
    for db in $FRAPPE_DBS; do
        local COUNT=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $db -se "
            SELECT (SELECT COUNT(*) FROM tabUser WHERE name NOT IN ('Administrator', 'Guest')) +
                   (SELECT COUNT(*) FROM tabEmployee) +
                   (SELECT COUNT(*) FROM tabCompany) as total;
        " 2>/dev/null || echo "0")
        if [ "$COUNT" -gt "$MAX_COUNT" ]; then
            MAX_COUNT=$COUNT
            SITE_DB=$db
        fi
    done

    if [ -z "$SITE_DB" ]; then
        SITE_DB=$(echo "$FRAPPE_DBS" | head -1)
    fi

    log_info "Using Frappe database: $SITE_DB (with $MAX_COUNT records)"

    # Get users
    local USERS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'email', email,
                'full_name', full_name,
                'enabled', enabled,
                'user_type', user_type,
                'creation', creation
            )
        ), JSON_ARRAY()) FROM tabUser WHERE name NOT IN ('Administrator', 'Guest') ;
    " 2>/dev/null || echo '[]')

    # Get employees
    local EMPLOYEES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'employee_name', employee_name,
                'first_name', first_name,
                'last_name', last_name,
                'designation', designation,
                'department', department,
                'company', company,
                'status', status,
                'date_of_joining', date_of_joining,
                'user_id', user_id
            )
        ), JSON_ARRAY()) FROM tabEmployee ;
    " 2>/dev/null || echo '[]')

    # Get companies
    local COMPANIES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'company_name', company_name,
                'domain', domain,
                'country', country,
                'default_currency', default_currency
            )
        ), JSON_ARRAY()) FROM tabCompany ;
    " 2>/dev/null || echo '[]')

    # Get departments
    local DEPARTMENTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'department_name', department_name,
                'parent_department', parent_department,
                'company', company
            )
        ), JSON_ARRAY()) FROM tabDepartment ;
    " 2>/dev/null || echo '[]')

    # Get attendance records
    local ATTENDANCE=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'employee', employee,
                'employee_name', employee_name,
                'status', status,
                'attendance_date', attendance_date,
                'company', company
            )
        ), JSON_ARRAY()) FROM tabAttendance ;
    " 2>/dev/null || echo '[]')

    # Get leave applications
    local LEAVES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'employee', employee,
                'employee_name', employee_name,
                'leave_type', leave_type,
                'from_date', from_date,
                'to_date', to_date,
                'total_leave_days', total_leave_days,
                'status', status
            )
        ), JSON_ARRAY()) FROM \`tabLeave Application\` ;
    " 2>/dev/null || echo '[]')

    # Get salary structures
    local SALARY_STRUCTURES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'company', company,
                'is_active', is_active,
                'payroll_frequency', payroll_frequency
            )
        ), JSON_ARRAY()) FROM \`tabSalary Structure\` ;
    " 2>/dev/null || echo '[]')

    # Get shift types
    local SHIFT_TYPES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'start_time', start_time,
                'end_time', end_time
            )
        ), JSON_ARRAY()) FROM \`tabShift Type\` ;
    " 2>/dev/null || echo '[]')

    # Get holiday lists
    local HOLIDAY_LISTS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'holiday_list_name', holiday_list_name,
                'from_date', from_date,
                'to_date', to_date
            )
        ), JSON_ARRAY()) FROM \`tabHoliday List\` ;
    " 2>/dev/null || echo '[]')

    # Get leave types
    local LEAVE_TYPES=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'leave_type_name', leave_type_name,
                'max_leaves_allowed', max_leaves_allowed,
                'is_carry_forward', is_carry_forward
            )
        ), JSON_ARRAY()) FROM \`tabLeave Type\` ;
    " 2>/dev/null || echo '[]')

    # Get designations
    local DESIGNATIONS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'designation_name', designation_name
            )
        ), JSON_ARRAY()) FROM \`tabDesignation\` ;
    " 2>/dev/null || echo '[]')

    # Get appraisals
    local APPRAISALS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT COALESCE(JSON_ARRAYAGG(
            JSON_OBJECT(
                'name', name,
                'employee', employee,
                'employee_name', employee_name,
                'start_date', start_date,
                'end_date', end_date,
                'status', status
            )
        ), JSON_ARRAY()) FROM \`tabAppraisal\` ;
    " 2>/dev/null || echo '[]')

    # Get statistics
    local STATS=$(docker exec $DB_CONTAINER $MYSQL_CMD -u$DB_USER -p$DB_PASS $SITE_DB -se "
        SELECT JSON_OBJECT(
            'total_users', (SELECT COUNT(*) FROM tabUser WHERE name NOT IN ('Administrator', 'Guest')),
            'total_employees', (SELECT COUNT(*) FROM tabEmployee),
            'total_companies', (SELECT COUNT(*) FROM tabCompany),
            'total_departments', (SELECT COUNT(*) FROM tabDepartment),
            'total_attendance', (SELECT COUNT(*) FROM tabAttendance),
            'total_leave_applications', (SELECT COUNT(*) FROM \`tabLeave Application\`),
            'total_salary_structures', (SELECT COUNT(*) FROM \`tabSalary Structure\`),
            'total_shift_types', (SELECT COUNT(*) FROM \`tabShift Type\`),
            'total_leave_types', (SELECT COUNT(*) FROM \`tabLeave Type\`)
        );
    " 2>/dev/null || echo '{}')

    echo "{
        \"service\": \"frappe_hrms\",
        \"url\": \"http://localhost:8084\",
        \"database\": \"mariadb\",
        \"site_database\": \"$SITE_DB\",
        \"data\": {
            \"statistics\": $STATS,
            \"users\": $USERS,
            \"employees\": $EMPLOYEES,
            \"companies\": $COMPANIES,
            \"departments\": $DEPARTMENTS,
            \"attendance\": $ATTENDANCE,
            \"leave_applications\": $LEAVES,
            \"salary_structures\": $SALARY_STRUCTURES,
            \"shift_types\": $SHIFT_TYPES,
            \"holiday_lists\": $HOLIDAY_LISTS,
            \"leave_types\": $LEAVE_TYPES,
            \"designations\": $DESIGNATIONS,
            \"appraisals\": $APPRAISALS
        },
        \"timestamp\": \"$(date -Iseconds)\"
    }" | format_json > "$OUTPUT_FILE"

    log_info "Frappe data saved to $OUTPUT_FILE"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting database data fetch from EnterpriseLab applications..."
    log_info "Output directory: $OUTPUT_DIR"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Check if python3 is available for JSON formatting
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is required but not installed."
        exit 1
    fi

    # Fetch data from all services
    fetch_rocketchat_data || log_warn "RocketChat database fetch failed"
    echo ""

    fetch_owncloud_data || log_warn "OwnCloud database fetch failed"
    echo ""

    fetch_gitlab_data || log_warn "GitLab database fetch failed"
    echo ""

    fetch_plane_data || log_warn "Plane database fetch failed"
    echo ""

    fetch_zammad_data || log_warn "Zammad database fetch failed"
    echo ""

    fetch_dolibarr_data || log_warn "Dolibarr database fetch failed"
    echo ""

    fetch_frappe_data || log_warn "Frappe database fetch failed"
    echo ""

    # Create summary
    log_info "Creating summary..."

    local SUMMARY_FILE="$OUTPUT_DIR/summary_database.json"

    echo "{
        \"fetch_timestamp\": \"$(date -Iseconds)\",
        \"fetch_method\": \"Direct Database Access\",
        \"script_location\": \"$SCRIPT_DIR\",
        \"output_directory\": \"$OUTPUT_DIR\",
        \"services\": {
            \"rocketchat\": {
                \"file\": \"rocketchat.json\",
                \"database\": \"mongodb\",
                \"app_url\": \"http://localhost:3000\"
            },
            \"owncloud\": {
                \"file\": \"owncloud.json\",
                \"database\": \"mariadb\",
                \"app_url\": \"http://localhost:8081\"
            },
            \"gitlab\": {
                \"file\": \"gitlab.json\",
                \"database\": \"postgresql\",
                \"app_url\": \"http://localhost:8080\"
            },
            \"plane\": {
                \"file\": \"plane.json\",
                \"database\": \"postgresql\",
                \"app_url\": \"http://localhost:80\"
            },
            \"zammad\": {
                \"file\": \"zammad.json\",
                \"database\": \"postgresql\",
                \"app_url\": \"http://localhost:8083\"
            },
            \"dolibarr\": {
                \"file\": \"dolibarr.json\",
                \"database\": \"mariadb\",
                \"app_url\": \"http://localhost:8082\"
            },
            \"frappe\": {
                \"file\": \"frappe.json\",
                \"database\": \"mariadb\",
                \"app_url\": \"http://localhost:8084\"
            }
        }
    }" | format_json > "$SUMMARY_FILE"

    log_info "Summary saved to $SUMMARY_FILE"
    echo ""
    log_info "✓ Database data fetch complete!"
    log_info "All JSON files with actual application data are available in: $OUTPUT_DIR"
    echo ""
    log_info "Files created:"
    ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | grep -v "summary.json" || log_warn "No JSON files were created"
}

# Run main function
main "$@"
