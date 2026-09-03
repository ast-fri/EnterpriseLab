#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_NAME=frappe-seeded
readonly ADMIN_USERNAME=admin
readonly ADMIN_PASSWORD='Admin123!'

die() { echo "frappe-seed: error: $*" >&2; exit 1; }
[[ $# -eq 2 ]] || die "Usage: $0 <frappe-json> <docker-compose.yml>"
command -v docker >/dev/null || die 'docker is required'
docker compose version >/dev/null || die 'Docker Compose v2 is required'

JSON_PATH="$(realpath -- "$1")"
COMPOSE_PATH="$(realpath -- "$2")"
EXPECTED_JSON="$(realpath -- "$(dirname -- "$COMPOSE_PATH")/../Extracted_data/frappe_from_db.json")"
[[ -r "$JSON_PATH" ]] || die "cannot read $JSON_PATH"
[[ -r "$COMPOSE_PATH" ]] || die "cannot read $COMPOSE_PATH"
[[ "$JSON_PATH" == "$EXPECTED_JSON" ]] || die "compose mounts $EXPECTED_JSON; pass that JSON file"

compose=(docker compose -p "$PROJECT_NAME" -f "$COMPOSE_PATH")
trap 'code=$?; if ((code)); then ${compose[@]} ps -a >&2 || true; ${compose[@]} logs --tail 180 frappe mariadb redis >&2 || true; fi; exit $code' EXIT

echo "frappe-seed: resetting isolated project $PROJECT_NAME"
"${compose[@]}" down -v --remove-orphans
echo 'frappe-seed: building the persistent v15 bench, importing JSON data, and creating task fixtures'
"${compose[@]}" up -d

ready=false
for _ in $(seq 1 720); do
  if curl -fsS --max-time 10 "http://127.0.0.1:${FRAPPE_HOST_PORT:-8084}/api/method/ping" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
[[ "$ready" == true ]] || die 'Frappe did not become ready within 60 minutes'

login_response="$(curl -fsS --max-time 20 -X POST \
  --data-urlencode "usr=$ADMIN_USERNAME" \
  --data-urlencode "pwd=$ADMIN_PASSWORD" \
  "http://127.0.0.1:${FRAPPE_HOST_PORT:-8084}/api/method/login")"
grep -q 'Logged In' <<<"$login_response" || die 'preset admin login failed'

cat <<EOF

Frappe HRMS was seeded successfully.
UI:          http://localhost:${FRAPPE_HOST_PORT:-8084}
Username:    $ADMIN_USERNAME
Password:    $ADMIN_PASSWORD
Credentials: $(dirname -- "$COMPOSE_PATH")/../user-credentials.json

Plain Docker Compose starts reuse this seeded site.
Run seed.sh again only to reset to the JSON and dummy-data baseline.
EOF
