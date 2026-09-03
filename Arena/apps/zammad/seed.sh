#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_NAME='zammad-seeded'
readonly ADMIN_USERNAME='admin'
readonly ADMIN_PASSWORD='Admin123!'

usage() { echo "Usage: $0 <zammad-json> <docker-compose.yml>" >&2; }
die() { echo "zammad-seed: error: $*" >&2; exit 1; }

[[ $# -eq 2 ]] || { usage; exit 2; }
command -v docker >/dev/null 2>&1 || die 'docker is required'
docker compose version >/dev/null 2>&1 || die 'Docker Compose v2 is required'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v curl >/dev/null 2>&1 || die 'curl is required'

readonly JSON_PATH="$(realpath -- "$1")"
readonly COMPOSE_PATH="$(realpath -- "$2")"
readonly EXPECTED_JSON_PATH="$(realpath -- "$(dirname -- "$COMPOSE_PATH")/../Extracted_data/zammad_from_db.json")"
[[ -r "$JSON_PATH" ]] || die "cannot read JSON file: $JSON_PATH"
[[ -r "$COMPOSE_PATH" ]] || die "cannot read compose file: $COMPOSE_PATH"
[[ "$JSON_PATH" == "$EXPECTED_JSON_PATH" ]] || die "compose mounts $EXPECTED_JSON_PATH; pass that JSON file"

compose=(docker compose -p "$PROJECT_NAME" -f "$COMPOSE_PATH")
trap 'status=$?; if ((status)); then ${compose[@]} ps -a >&2 || true; ${compose[@]} logs --tail 150 zammad-init zammad-seed zammad-postgresql zammad-railsserver >&2 || true; fi; exit $status' EXIT

echo "zammad-seed: resetting isolated Compose project '$PROJECT_NAME'"
"${compose[@]}" down -v --remove-orphans
echo 'zammad-seed: starting Zammad and importing the JSON baseline'
"${compose[@]}" up -d

seed_id=''
for _ in $(seq 1 180); do
  seed_id="$("${compose[@]}" ps -aq zammad-seed)"
  [[ -n "$seed_id" ]] && break
  sleep 5
done
[[ -n "$seed_id" ]] || die 'seed container was not created'

while true; do
  state="$(docker inspect -f '{{.State.Status}}' "$seed_id")"
  if [[ "$state" == exited ]]; then
    code="$(docker inspect -f '{{.State.ExitCode}}' "$seed_id")"
    "${compose[@]}" logs zammad-seed
    [[ "$code" == 0 ]] || die "seed container exited with code $code"
    break
  fi
  sleep 10
done

ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 10 "http://127.0.0.1:${ZAMMAD_HOST_PORT:-8050}/" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
[[ "$ready" == true ]] || die 'Zammad UI did not become ready within 15 minutes'
curl -fsS --max-time 20 "http://127.0.0.1:${ZAMMAD_HOST_PORT:-8050}/api/v1/users/me" \
  -u "$ADMIN_USERNAME:$ADMIN_PASSWORD" >/dev/null \
  || die 'Zammad UI is ready but the preset admin credential was rejected'

echo
echo 'Zammad was seeded successfully.'
echo "UI:          http://localhost:${ZAMMAD_HOST_PORT:-8050}"
echo "Username:    $ADMIN_USERNAME"
echo "Password:    $ADMIN_PASSWORD"
echo "Credentials: $(dirname -- "$COMPOSE_PATH")/../user-credentials.json"
echo
echo "Plain 'docker compose up -d' reuses this seeded database."
echo 'Run seed.sh again only when you want to reset to the JSON baseline.'
