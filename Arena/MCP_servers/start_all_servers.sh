#!/bin/bash

# Directory containing all the server folders
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find all subdirectories containing docker-compose.yml
SERVERS=( "$BASE_DIR"/* )

# Function to start all servers
start_all() {
  echo "🚀 Starting all servers..."
  for dir in "${SERVERS[@]}"; do
    if [ -f "$dir/docker-compose.yml" ]; then
      echo "➡️  Starting server in: $(basename "$dir")"
      (cd "$dir" && docker compose up -d)
    fi
  done
  echo "✅ All servers started successfully."
}

# Function to stop all servers
stop_all() {
  echo "🛑 Stopping all servers..."
  for dir in "${SERVERS[@]}"; do
    if [ -f "$dir/docker-compose.yml" ]; then
      echo "➡️  Stopping server in: $(basename "$dir")"
      (cd "$dir" && docker compose down)
    fi
  done
  echo "✅ All servers stopped successfully."
}

# Main logic
case "$1" in
  stop)
    stop_all
    ;;
  *)
    start_all
    ;;
esac
