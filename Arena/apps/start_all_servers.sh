#!/bin/bash
# ==============================================================
# Enterprise Lab – Multi-App Launcher (env-aware version)
# ==============================================================

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

check_dependencies() {
    echo "🔍 Checking Docker installation..."
    if ! command -v docker &>/dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        exit 1
    fi

    echo "🔍 Checking Docker Compose..."
    if docker compose version &>/dev/null; then
        COMPOSE_CMD="docker compose"
    elif docker-compose version &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        echo "❌ Docker Compose not found. Please install Docker Compose v1 or v2."
        exit 1
    fi

    echo "✅ Dependencies OK"
}

find_compose_files() {
    echo "🔍 Scanning for compose files..."
    mapfile -t COMPOSE_FILES < <(
        find "$ROOT_DIR" -maxdepth 3 \
            \( -name "docker-compose.yml" -o -name "docker-compose.yaml" -o -name "compose.yml" -o -name "compose.yaml" -o -name "*-compose.yml" -o -name "*-compose.yaml" \) \
            | sort
    )

    if [ ${#COMPOSE_FILES[@]} -eq 0 ]; then
        echo "⚠️  No docker-compose files found under $ROOT_DIR"
        exit 1
    fi

    echo "✅ Found ${#COMPOSE_FILES[@]} compose files:"
    for cf in "${COMPOSE_FILES[@]}"; do
        echo "   - $(realpath --relative-to="$ROOT_DIR" "$cf")"
    done
}


start_all() {
    echo "🚀 Starting all enterprise servers..."
    for compose_file in "${COMPOSE_FILES[@]}"; do
        APP_DIR=$(dirname "$compose_file")
        APP_NAME=$(basename "$APP_DIR")

        # Look for .env or *env file
        ENV_FILE=$(find "$APP_DIR" -maxdepth 1 -type f -name "*.env" | head -n 1)

        echo "➡️  Starting $APP_NAME ..."
        if [ -n "$ENV_FILE" ]; then
            echo "   Using env file: $(basename "$ENV_FILE")"
            (cd "$APP_DIR" && $COMPOSE_CMD --env-file "$(basename "$ENV_FILE")" up -d) | tee "$LOG_DIR/$APP_NAME.log"
        else
            (cd "$APP_DIR" && $COMPOSE_CMD up -d) | tee "$LOG_DIR/$APP_NAME.log"
        fi
    done
    echo "✅ All servers started successfully!"
}

stop_all() {
    echo "🛑 Stopping all enterprise servers..."
    for compose_file in "${COMPOSE_FILES[@]}"; do
        APP_DIR=$(dirname "$compose_file")
        APP_NAME=$(basename "$APP_DIR")
        echo "➡️  Stopping $APP_NAME ..."
        (cd "$APP_DIR" && $COMPOSE_CMD down)
    done
    echo "✅ All servers stopped successfully!"
}

status_all() {
    echo "📊 Status of all enterprise servers:"
    for compose_file in "${COMPOSE_FILES[@]}"; do
        APP_DIR=$(dirname "$compose_file")
        APP_NAME=$(basename "$APP_DIR")
        echo "------------------------------------------"
        echo "📦 $APP_NAME"
        (cd "$APP_DIR" && $COMPOSE_CMD ps)
    done
}

# ---- main ----
check_dependencies
find_compose_files

ACTION=${1:-start}
case "$ACTION" in
    start) start_all ;;
    stop) stop_all ;;
    status) status_all ;;
    *) echo "Usage: $0 {start|stop|status}" && exit 1 ;;
esac
