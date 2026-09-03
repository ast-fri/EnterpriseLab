#!/usr/bin/env bash
set -Eeuo pipefail

readonly imported_marker='/opt/zammad/storage/.enterprise-arena-json-imported'
readonly complete_marker='/opt/zammad/storage/.enterprise-arena-seeded'

if [[ -f "$complete_marker" ]]; then
  echo 'zammad-seed: complete marker exists; preserving the current database'
  exit 0
fi

echo 'zammad-seed: loading JSON and applying shared credentials'
rm -f /tmp/zammad-seed-imported
bundle exec rails runner /seed/import-zammad.rb

if [[ -f "$imported_marker" ]]; then
  echo 'zammad-seed: rebuilding the Elasticsearch index'
  bundle exec rake zammad:searchindex:rebuild
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$complete_marker"
fi
