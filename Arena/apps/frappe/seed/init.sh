#!/usr/bin/env bash
set -Eeuo pipefail

readonly BENCH_DIR=/home/frappe/frappe-bench
readonly SITE_NAME="${FRAPPE_SITE:-hrms.localhost}"
readonly SEED_CACHE_DIR="$BENCH_DIR/.seed-cache"

export YARN_CACHE_FOLDER="$SEED_CACHE_DIR/yarn"
export PIP_CACHE_DIR="$SEED_CACHE_DIR/pip"

if [[ ! -d "$BENCH_DIR/apps/frappe" ]]; then
  echo 'frappe-init: creating a persistent Frappe v15 bench'
  bench init --ignore-exist --skip-redis-config-generation --frappe-branch version-15 "$BENCH_DIR"
fi

cd "$BENCH_DIR"
mkdir -p /home/frappe/logs
mkdir -p "$BENCH_DIR/$SITE_NAME/logs"
if [[ -f sites/common_site_config.json && ! -s sites/common_site_config.json ]]; then
  echo 'frappe-init: repairing empty common_site_config.json after an interrupted write'
  printf '{}\n' > sites/common_site_config.json
fi
bench set-mariadb-host mariadb
bench set-redis-cache-host redis://redis:6379
bench set-redis-queue-host redis://redis:6379
bench set-redis-socketio-host redis://redis:6379
sed -i '/^redis_/d; /^watch:/d' Procfile

if [[ ! -d apps/erpnext ]]; then
  echo 'frappe-init: installing ERPNext v15 application code'
  bench get-app --branch version-15 erpnext
fi
if [[ ! -d apps/hrms ]]; then
  echo 'frappe-init: installing HRMS v15 application code'
  bench get-app --branch version-15 hrms
fi
rm -rf "$SEED_CACHE_DIR"

if [[ ! -f "sites/$SITE_NAME/site_config.json" ]]; then
  echo "frappe-init: creating site $SITE_NAME"
  bench new-site "$SITE_NAME" \
    --mariadb-user-host-login-scope='%' \
    --db-root-username=root \
    --db-root-password="${FRAPPE_DB_ROOT_PASSWORD:-frappe}" \
    --admin-password='Admin123!' \
    --install-app erpnext \
    --no-mariadb-socket
  bench --site "$SITE_NAME" install-app hrms
fi

bench use "$SITE_NAME"
if [[ ! -f "sites/$SITE_NAME/.enterprise-arena-seeded" ]]; then
  echo 'frappe-init: importing validated JSON records and dummy task data'
  FRAPPE_SITE="$SITE_NAME" ./env/bin/python /workspace/seed_frappe.py
fi

bench --site "$SITE_NAME" enable-scheduler
bench --site "$SITE_NAME" clear-cache
echo 'frappe-init: starting Frappe services'
exec bench start
