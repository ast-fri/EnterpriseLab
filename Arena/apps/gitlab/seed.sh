#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="${1:-$script_dir/docker-compose.yaml}"
seed_json="${2:-$script_dir/../Extracted_data/gitlab_from_db.json}"

if [[ ! -r "$compose_file" ]]; then
  echo "Compose file is not readable: $compose_file" >&2
  exit 1
fi

if [[ ! -r "$seed_json" ]]; then
  echo "Seed JSON is not readable: $seed_json" >&2
  exit 1
fi

expected_json="$(cd "$script_dir/.." && pwd)/Extracted_data/gitlab_from_db.json"
actual_json="$(cd "$(dirname "$seed_json")" && pwd)/$(basename "$seed_json")"
if [[ "$actual_json" != "$expected_json" ]]; then
  echo "This compose file mounts $expected_json; received $actual_json" >&2
  echo "Copy the desired export to the mounted path or update the compose bind mount." >&2
  exit 1
fi

echo "Removing only the isolated gitlab-seeded containers and volumes..."
docker compose -f "$compose_file" -p gitlab-seeded down --volumes --remove-orphans

echo "Starting a fresh GitLab instance. First boot and seeding can take 10-20 minutes..."
docker compose -f "$compose_file" -p gitlab-seeded up -d

echo "Waiting for the one-shot seed service..."
seed_container=""
for _ in $(seq 1 120); do
  seed_container="$(docker compose -f "$compose_file" -p gitlab-seeded ps -q seed)"
  [[ -n "$seed_container" ]] && break
  sleep 10
done

if [[ -z "$seed_container" ]]; then
  echo "The seed container was not created. Inspect: docker compose -f $compose_file -p gitlab-seeded logs" >&2
  exit 1
fi

while true; do
  status="$(docker inspect -f '{{.State.Status}}' "$seed_container")"
  if [[ "$status" == "exited" ]]; then
    exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$seed_container")"
    docker compose -f "$compose_file" -p gitlab-seeded logs seed
    [[ "$exit_code" == "0" ]] || exit "$exit_code"
    break
  fi
  sleep 10
done

curl -fsS --max-time 20 http://127.0.0.1:8081/users/sign_in >/dev/null
echo "GitLab is seeded and available at http://localhost:8081"
echo "Credentials: $script_dir/../user-credentials.json"
