#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_NAME='rocketchat-seeded'
readonly ADMIN_USERNAME='admin'
readonly ADMIN_PASSWORD='Admin123!'

usage() {
  echo "Usage: $0 <rocketchat-json> <docker-compose.yml>" >&2
}

die() {
  echo "seed: error: $*" >&2
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

command -v docker >/dev/null 2>&1 || die 'docker is required'
docker compose version >/dev/null 2>&1 || die 'Docker Compose v2 is required'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'

readonly JSON_PATH="$(realpath -- "$1")"
readonly COMPOSE_PATH="$(realpath -- "$2")"
readonly EXPECTED_JSON_PATH="$(realpath -- "$(dirname -- "$COMPOSE_PATH")/../Extracted_data/rocketchat_from_db.json")"

[[ -r "$JSON_PATH" ]] || die "cannot read JSON file: $JSON_PATH"
[[ -r "$COMPOSE_PATH" ]] || die "cannot read compose file: $COMPOSE_PATH"
[[ "$JSON_PATH" == "$EXPECTED_JSON_PATH" ]] \
  || die "compose currently mounts $EXPECTED_JSON_PATH; pass that JSON file"

export ROCKETCHAT_SEED_MODE=reset
compose=(docker compose -p "$PROJECT_NAME" -f "$COMPOSE_PATH")

show_failure_context() {
  local status=$?
  if (( status != 0 )); then
    echo 'seed: failed; recent container state and logs follow' >&2
    "${compose[@]}" ps >&2 || true
    "${compose[@]}" logs --tail 100 mongodb seed credentials nats rocketchat >&2 || true
  fi
  exit "$status"
}
trap show_failure_context ERR

echo "seed: resetting isolated Compose project '${PROJECT_NAME}'"
"${compose[@]}" down -v --remove-orphans

echo 'seed: starting MongoDB, NATS, automatic import, and Rocket.Chat'
"${compose[@]}" up -d

container_port="${PORT:-3000}"
echo 'seed: waiting for the Rocket.Chat API'
api_ready=false
for _ in $(seq 1 180); do
  if "${compose[@]}" exec -T rocketchat node -e \
    "fetch('http://127.0.0.1:${container_port}/api/info').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))" \
    >/dev/null 2>&1; then
    api_ready=true
    break
  fi
  sleep 2
done
[[ "$api_ready" == true ]] || die 'Rocket.Chat did not become ready within 360 seconds'

echo 'seed: verifying representative imported data'
"${compose[@]}" exec -T mongodb mongosh --quiet \
  'mongodb://localhost:27017/rocketchat?replicaSet=rs0' \
  --eval '
    const checks = {
      users: db.users.countDocuments({_id: {$ne: "rocket.cat"}}),
      rooms: db.rocketchat_room.countDocuments({}),
      messages: db.rocketchat_message.countDocuments({}),
      subscriptions: db.rocketchat_subscription.countDocuments({}),
      avatarChunks: db.getCollection("rocketchat_avatars.chunks").countDocuments({}),
      seedMarker: db.rocketchat_seed_metadata.countDocuments({_id: "json-baseline"}),
    };
    if (checks.users < 13 || checks.rooms < 1 || checks.messages < 1 || checks.subscriptions < 1 || checks.avatarChunks < 1 || checks.seedMarker !== 1) {
      printjson(checks);
      quit(1);
    }
    print(`seed: database checks passed ${JSON.stringify(checks)}`);
  '

echo 'seed: verifying UI credentials through the Rocket.Chat login API'
"${compose[@]}" exec -T \
  -e "SEED_ADMIN_USERNAME=${ADMIN_USERNAME}" \
  -e "SEED_ADMIN_PASSWORD=${ADMIN_PASSWORD}" \
  rocketchat node -e "
    fetch('http://127.0.0.1:${container_port}/api/v1/login', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({user: process.env.SEED_ADMIN_USERNAME, password: process.env.SEED_ADMIN_PASSWORD}),
    }).then(async (response) => {
      const body = await response.json();
      if (!response.ok || body.status !== 'success' || !body.data?.authToken) {
        console.error(JSON.stringify(body));
        process.exit(1);
      }
    }).catch((error) => { console.error(error); process.exit(1); });
  "

published_address="$("${compose[@]}" port rocketchat "$container_port" | head -n 1)"
published_port="${published_address##*:}"

cat <<EOF

Rocket.Chat was seeded successfully.
UI:       http://localhost:${published_port}
Username: ${ADMIN_USERNAME}
Password: ${ADMIN_PASSWORD}

Shared credentials: $(dirname -- "$COMPOSE_PATH")/../user-credentials.json
Plain 'docker compose up -d' now reuses this seeded database.
Run seed.sh again only when you want to reset to the JSON baseline.
EOF
