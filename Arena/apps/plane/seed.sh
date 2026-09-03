#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_NAME=plane-seeded
readonly ADMIN_EMAIL=admin@inazuma.com
readonly ADMIN_PASSWORD='Admin123!'

die() { echo "plane-seed: error: $*" >&2; exit 1; }
[[ $# -eq 2 ]] || die "Usage: $0 <plane-json> <docker-compose.yaml>"
command -v docker >/dev/null || die 'docker is required'
docker compose version >/dev/null || die 'Docker Compose v2 is required'

JSON_PATH="$(realpath -- "$1")"
COMPOSE_PATH="$(realpath -- "$2")"
COMPOSE_DIR="$(dirname -- "$COMPOSE_PATH")"
ENV_PATH="$COMPOSE_DIR/plane.env"
EXPECTED_JSON="$(realpath -- "$COMPOSE_DIR/../Extracted_data/plane_from_db.json")"
[[ -r "$JSON_PATH" ]] || die "cannot read $JSON_PATH"
[[ -r "$COMPOSE_PATH" ]] || die "cannot read $COMPOSE_PATH"
[[ -r "$ENV_PATH" ]] || die "cannot read $ENV_PATH"
[[ "$JSON_PATH" == "$EXPECTED_JSON" ]] || die "compose mounts $EXPECTED_JSON; pass that JSON file"

compose=(docker compose -p "$PROJECT_NAME" --env-file "$ENV_PATH" -f "$COMPOSE_PATH")
trap 'code=$?; if ((code)); then ${compose[@]} ps -a >&2 || true; ${compose[@]} logs --tail 200 seed migrator api plane-db >&2 || true; fi; exit $code' EXIT

echo "plane-seed: resetting isolated project $PROJECT_NAME"
"${compose[@]}" down -v --remove-orphans
echo 'plane-seed: starting Plane, migrating its schema, and restoring the JSON baseline'
"${compose[@]}" up -d

ready=false
for _ in $(seq 1 360); do
  if curl -fsS --max-time 10 "http://127.0.0.1:${LISTEN_HTTP_PORT:-3001}/" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
[[ "$ready" == true ]] || die 'Plane did not become ready within 30 minutes'

seed_status="$("${compose[@]}" ps --status exited --format json seed)"
grep -q '"ExitCode":0' <<<"$seed_status" || die 'Plane seed service did not finish successfully'

cat <<EOF

Plane was seeded successfully.
UI:          http://localhost:${LISTEN_HTTP_PORT:-3001}
Admin email: $ADMIN_EMAIL
Password:    $ADMIN_PASSWORD
Credentials: $COMPOSE_DIR/../user-credentials.json

Plain Docker Compose starts reuse this seeded database.
Run seed.sh again only to reset to the JSON baseline.
EOF
