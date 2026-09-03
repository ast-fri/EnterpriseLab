#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_NAME='dolibarr-seeded'
readonly ADMIN_USERNAME='admin'
readonly ADMIN_PASSWORD='Admin123!'

usage() {
  echo "Usage: $0 <dolibarr-json> <docker-compose.yml>" >&2
}

die() {
  echo "dolibarr-seed: error: $*" >&2
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

command -v docker >/dev/null 2>&1 || die 'docker is required'
docker compose version >/dev/null 2>&1 || die 'Docker Compose v2 is required'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v curl >/dev/null 2>&1 || die 'curl is required'

readonly JSON_PATH="$(realpath -- "$1")"
readonly COMPOSE_PATH="$(realpath -- "$2")"
readonly EXPECTED_JSON_PATH="$(realpath -- "$(dirname -- "$COMPOSE_PATH")/../Extracted_data/dolibarr_from_db.json")"

[[ -r "$JSON_PATH" ]] || die "cannot read JSON file: $JSON_PATH"
[[ -r "$COMPOSE_PATH" ]] || die "cannot read compose file: $COMPOSE_PATH"
[[ "$JSON_PATH" == "$EXPECTED_JSON_PATH" ]] \
  || die "compose currently mounts $EXPECTED_JSON_PATH; pass that JSON file"

compose=(docker compose -p "$PROJECT_NAME" -f "$COMPOSE_PATH")

show_failure_context() {
  local status=$?
  if (( status != 0 )); then
    echo 'dolibarr-seed: failed; recent container state and logs follow' >&2
    "${compose[@]}" ps -a >&2 || true
    "${compose[@]}" logs --tail 150 mariadb web >&2 || true
  fi
  exit "$status"
}
trap show_failure_context ERR

echo "dolibarr-seed: resetting isolated Compose project '${PROJECT_NAME}'"
"${compose[@]}" down -v --remove-orphans

echo 'dolibarr-seed: starting MariaDB and Dolibarr; first boot imports the JSON automatically'
"${compose[@]}" up -d

ready=false
for _ in $(seq 1 180); do
  health="$("${compose[@]}" ps --format json web 2>/dev/null \
    | sed -n 's/.*"Health":"\([^"]*\)".*/\1/p' | head -n 1)"
  if [[ "$health" == 'healthy' ]]; then
    ready=true
    break
  fi
  sleep 5
done
[[ "$ready" == true ]] || die 'Dolibarr did not become healthy within 15 minutes'

echo 'dolibarr-seed: verifying imported records and preset users'
"${compose[@]}" exec -T mariadb mariadb \
  --user="${MYSQL_USER:-dolidbuser}" \
  --password="${MYSQL_PASSWORD:-dolidbpass}" \
  "${MYSQL_DATABASE:-dolidb}" \
  --batch --skip-column-names \
  --execute="
    SELECT IF(COUNT(*) = 13, 'credential-users=13', CONCAT('credential-users=', COUNT(*)))
    FROM llx_user
    WHERE statut = 1
      AND login IN (
        'admin', 'abigail.mitchell', 'aarav.mittal', 'surya.reddy',
        'raj.patel', 'rahul.khanna', 'karan.sharma', 'priya.arora',
        'sameer.malhotra', 'ethan.reynolds', 'anjali.mathew',
        'vandana.reddy', 'neeraj.sharma'
      );
    SELECT CONCAT('projects=', COUNT(*)) FROM llx_projet;
    SELECT CONCAT('companies=', COUNT(*)) FROM llx_societe;
    SELECT CONCAT('products=', COUNT(*)) FROM llx_product;
    SELECT CONCAT('invoices=', COUNT(*)) FROM llx_facture;
  "

curl -fsS --max-time 20 http://127.0.0.1:${DOLIBARR_HOST_PORT:-8082}/index.php >/dev/null

cat <<EOF

Dolibarr was seeded successfully.
UI:          http://localhost:${DOLIBARR_HOST_PORT:-8082}
Username:    ${ADMIN_USERNAME}
Password:    ${ADMIN_PASSWORD}
Credentials: $(dirname -- "$COMPOSE_PATH")/../user-credentials.json

Plain 'docker compose up -d' now reuses this seeded database.
Run seed.sh again only when you want to reset to the JSON baseline.
EOF
