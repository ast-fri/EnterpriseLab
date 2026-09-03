#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_NAME=owncloud-seeded
readonly HOST_PORT="${OWNCLOUD_HOST_PORT:-8083}"

die() { echo "owncloud-seed: error: $*" >&2; exit 1; }
[[ $# -eq 2 ]] || die "Usage: $0 <owncloud-json> <docker-compose.yml>"
command -v docker >/dev/null || die 'docker is required'
command -v python3 >/dev/null || die 'python3 is required'
command -v curl >/dev/null || die 'curl is required'
docker compose version >/dev/null || die 'Docker Compose v2 is required'

JSON_PATH="$(realpath -- "$1")"
COMPOSE_PATH="$(realpath -- "$2")"
COMPOSE_DIR="$(dirname -- "$COMPOSE_PATH")"
CREDENTIALS_PATH="$COMPOSE_DIR/../user-credentials.json"
SEEDER_PATH="$COMPOSE_DIR/seed_owncloud.py"
[[ -r "$JSON_PATH" ]] || die "cannot read $JSON_PATH"
[[ -r "$COMPOSE_PATH" ]] || die "cannot read $COMPOSE_PATH"
[[ -r "$CREDENTIALS_PATH" ]] || die "cannot read $CREDENTIALS_PATH"
[[ -r "$SEEDER_PATH" ]] || die "cannot read $SEEDER_PATH"

export OWNCLOUD_HOST_PORT="$HOST_PORT"
compose=(docker compose -p "$PROJECT_NAME" -f "$COMPOSE_PATH")
trap 'code=$?; if ((code)); then ${compose[@]} ps -a >&2 || true; ${compose[@]} logs --tail 200 owncloud db >&2 || true; fi; exit $code' EXIT

echo "owncloud-seed: resetting isolated project $PROJECT_NAME"
"${compose[@]}" down -v --remove-orphans
echo 'owncloud-seed: starting a clean ownCloud database'
"${compose[@]}" up -d

ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 10 "http://127.0.0.1:$HOST_PORT/status.php" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
[[ "$ready" == true ]] || die 'ownCloud did not become ready within 15 minutes'

echo 'owncloud-seed: creating users and reconstructing exported file paths'
python3 "$SEEDER_PATH" \
  --json "$JSON_PATH" \
  --credentials "$CREDENTIALS_PATH" \
  --url "http://127.0.0.1:$HOST_PORT"

cat <<EOF

ownCloud was seeded successfully.
UI:          http://localhost:$HOST_PORT
Admin user:  admin
Password:    Admin123!
Credentials: $CREDENTIALS_PATH

Plain Docker Compose starts reuse this seeded database.
Run seed.sh again only when you want to reset to the JSON baseline.
EOF
