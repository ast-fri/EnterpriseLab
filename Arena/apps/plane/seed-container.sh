#!/usr/bin/env bash
set -Eeuo pipefail

readonly MARKER=/seed-state/seeded

if [[ -f "$MARKER" ]]; then
  echo 'plane-seed: existing seeded baseline found; skipping import'
  exit 0
fi

cd /code
python /seed/seed_plane.py
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$MARKER"
echo 'plane-seed: marker written; ordinary Compose starts will preserve this state'
