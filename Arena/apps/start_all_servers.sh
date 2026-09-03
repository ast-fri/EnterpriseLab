#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="$ROOT_DIR/logs"

readonly -a APPS=(rocketchat gitlab dolibarr zammad frappe plane owncloud)

declare -Ar PROJECT=(
  [rocketchat]='rocketchat-seeded'
  [gitlab]='gitlab-seeded'
  [dolibarr]='dolibarr-seeded'
  [zammad]='zammad-seeded'
  [frappe]='frappe-seeded'
  [plane]='plane-seeded'
  [owncloud]='owncloud-seeded'
)

declare -Ar COMPOSE_FILE=(
  [rocketchat]='rocketchat/docker-compose.yml'
  [gitlab]='gitlab/docker-compose.yaml'
  [dolibarr]='dolibarr/docker-compose.yml'
  [zammad]='zammad/docker-compose.yml'
  [frappe]='frappe/docker-compose.yml'
  [plane]='plane/docker-compose.yaml'
  [owncloud]='owncloud/docker-compose.yml'
)

declare -Ar ENV_FILE=(
  [rocketchat]=''
  [gitlab]=''
  [dolibarr]=''
  [zammad]=''
  [frappe]=''
  [plane]='plane/plane.env'
  [owncloud]=''
)

# Deliberately excludes one-shot import, migration, credential, and init services.
declare -Ar RUNTIME_SERVICES=(
  [rocketchat]='mongodb nats rocketchat'
  [gitlab]='gitlab'
  [dolibarr]='mariadb web'
  [zammad]='zammad-memcached zammad-postgresql zammad-railsserver zammad-nginx zammad-websocket zammad-scheduler zammad-elasticsearch zammad-redis'
  [frappe]='mariadb redis frappe'
  [plane]='plane-db plane-mq plane-redis plane-minio api worker web space beat-worker live admin proxy'
  [owncloud]='db owncloud'
)

declare -Ar URL=(
  [rocketchat]='http://localhost:3000'
  [gitlab]='http://localhost:8081'
  [dolibarr]='http://localhost:8082'
  [zammad]='http://localhost:8050'
  [frappe]='http://localhost:8084'
  [plane]='http://localhost:3001'
  [owncloud]='http://localhost:8083'
)

usage() {
  cat <<'EOF'
Usage: ./start_all_servers.sh [MODE]

Modes:
  --continue  Start/recreate runtime containers and reuse current seeded data.
              This is the default when no mode is supplied.
  --stop      Stop all runtime containers without deleting containers or data.
  --reset     Delete each seeded project's volumes and rebuild its baseline.
  --status    Show Compose status and application URLs.
  --help      Show this help.

Short aliases without leading dashes are also accepted: continue, start, stop,
reset, status, and help.
EOF
}

die() {
  echo "all-servers: error: $*" >&2
  exit 1
}

check_dependencies() {
  command -v docker >/dev/null 2>&1 || die 'docker is required'
  docker compose version >/dev/null 2>&1 || die 'Docker Compose v2 is required'
  mkdir -p "$LOG_DIR"
  local app path
  for app in "${APPS[@]}"; do
    path="$ROOT_DIR/${COMPOSE_FILE[$app]}"
    [[ -r "$path" ]] || die "missing Compose file: $path"
    if [[ -n "${ENV_FILE[$app]}" ]]; then
      path="$ROOT_DIR/${ENV_FILE[$app]}"
      [[ -r "$path" ]] || die "missing environment file: $path"
    fi
  done
}

# Populates the global COMPOSE array for the requested application.
compose_for() {
  local app="$1"
  COMPOSE=(docker compose -p "${PROJECT[$app]}")
  if [[ -n "${ENV_FILE[$app]}" ]]; then
    COMPOSE+=(--env-file "$ROOT_DIR/${ENV_FILE[$app]}")
  fi
  COMPOSE+=(-f "$ROOT_DIR/${COMPOSE_FILE[$app]}")
}

has_saved_state() {
  local app="$1" containers volumes
  compose_for "$app"
  containers="$("${COMPOSE[@]}" ps -aq 2>/dev/null || true)"
  volumes="$(docker volume ls -q --filter "label=com.docker.compose.project=${PROJECT[$app]}" 2>/dev/null || true)"
  [[ -n "$containers" || -n "$volumes" ]]
}

continue_app() {
  local app="$1"
  local -a services
  if ! has_saved_state "$app"; then
    echo "No saved state exists for $app. Run --reset once to create it." >&2
    return 1
  fi
  compose_for "$app"
  read -r -a services <<<"${RUNTIME_SERVICES[$app]}"
  echo "[$app] starting runtime services and reusing ${PROJECT[$app]} volumes"
  "${COMPOSE[@]}" up -d --no-deps "${services[@]}"
}

stop_app() {
  local app="$1"
  local -a services
  compose_for "$app"
  read -r -a services <<<"${RUNTIME_SERVICES[$app]}"
  echo "[$app] stopping runtime services; seeded data is preserved"
  "${COMPOSE[@]}" stop "${services[@]}"
}

reset_app() {
  local app="$1"
  echo "[$app] resetting and seeding from its extracted JSON baseline"
  case "$app" in
    rocketchat)
      "$ROOT_DIR/rocketchat/seed.sh" \
        "$ROOT_DIR/Extracted_data/rocketchat_from_db.json" \
        "$ROOT_DIR/rocketchat/docker-compose.yml"
      ;;
    gitlab)
      "$ROOT_DIR/gitlab/seed.sh" \
        "$ROOT_DIR/gitlab/docker-compose.yaml" \
        "$ROOT_DIR/Extracted_data/gitlab_from_db.json"
      ;;
    dolibarr)
      "$ROOT_DIR/dolibarr/seed.sh" \
        "$ROOT_DIR/Extracted_data/dolibarr_from_db.json" \
        "$ROOT_DIR/dolibarr/docker-compose.yml"
      ;;
    zammad)
      "$ROOT_DIR/zammad/seed.sh" \
        "$ROOT_DIR/Extracted_data/zammad_from_db.json" \
        "$ROOT_DIR/zammad/docker-compose.yml"
      ;;
    frappe)
      "$ROOT_DIR/frappe/seed/seed.sh" \
        "$ROOT_DIR/Extracted_data/frappe_from_db.json" \
        "$ROOT_DIR/frappe/docker-compose.yml"
      ;;
    plane)
      "$ROOT_DIR/plane/seed.sh" \
        "$ROOT_DIR/Extracted_data/plane_from_db.json" \
        "$ROOT_DIR/plane/docker-compose.yaml"
      ;;
    owncloud)
      "$ROOT_DIR/owncloud/seed.sh" \
        "$ROOT_DIR/Extracted_data/owncloud_from_db.json" \
        "$ROOT_DIR/owncloud/docker-compose.yml"
      ;;
    *) die "unknown application: $app" ;;
  esac
}

run_all() {
  local operation="$1" app log_path
  local -a failures=()
  for app in "${APPS[@]}"; do
    log_path="$LOG_DIR/${operation}-${app}.log"
    if "$operation"_app "$app" 2>&1 | tee "$log_path"; then
      :
    else
      failures+=("$app")
    fi
  done
  if ((${#failures[@]})); then
    die "$operation failed for: ${failures[*]} (see $LOG_DIR)"
  fi
}

status_all() {
  local app
  for app in "${APPS[@]}"; do
    compose_for "$app"
    printf '\n[%s] %s\n' "$app" "${URL[$app]}"
    "${COMPOSE[@]}" ps -a
  done
}

main() {
  [[ $# -le 1 ]] || { usage >&2; exit 2; }
  local mode="${1:---continue}"
  case "$mode" in
    --continue|continue|start|--start) mode='continue' ;;
    --stop|stop) mode='stop' ;;
    --reset|reset) mode='reset' ;;
    --status|status) mode='status' ;;
    --help|-h|help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac

  check_dependencies
  case "$mode" in
    continue)
      run_all continue
      echo
      echo 'All applications are starting with their current seeded state.'
      ;;
    stop)
      run_all stop
      echo
      echo 'All application containers are stopped; seeded data is preserved.'
      ;;
    reset)
      echo 'WARNING: --reset deletes and rebuilds all seven seeded app volumes.'
      echo 'The complete reset runs sequentially and can take well over an hour.'
      "$ROOT_DIR/sanitize_seed_exports.py" --check
      run_all reset
      echo
      echo 'All applications were reset and seeded successfully.'
      ;;
    status) status_all ;;
  esac

  if [[ "$mode" == continue || "$mode" == reset ]]; then
    printf '\nApplication URLs:\n'
    local app
    for app in "${APPS[@]}"; do printf '  %-12s %s\n' "$app" "${URL[$app]}"; done
  fi
}

main "$@"
