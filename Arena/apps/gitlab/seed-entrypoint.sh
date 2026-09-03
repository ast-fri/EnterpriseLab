#!/bin/bash
set -euo pipefail

seed_json="${SEED_JSON:-/seed-data/gitlab_from_db.json}"
credentials_json="${CREDENTIALS_JSON:-/seed/user-credentials.json}"
marker="/var/opt/gitlab/.json-seed-complete"

if [[ ! -r "$seed_json" ]]; then
  echo "Seed JSON is not readable: $seed_json" >&2
  exit 1
fi

if [[ ! -r "$credentials_json" ]]; then
  echo "Credentials JSON is not readable: $credentials_json" >&2
  exit 1
fi

seed_hash="$(sha256sum "$seed_json" | cut -d ' ' -f 1)"

if [[ -f "$marker" ]]; then
  existing_hash="$(tr -d '[:space:]' < "$marker")"
  if [[ "$existing_hash" != "$seed_hash" ]]; then
    echo "The seed JSON changed after this volume was initialized." >&2
    echo "Run ./seed.sh to rebuild the isolated seeded volumes." >&2
    exit 1
  fi
  export SKIP_DATA_IMPORT=1
fi

export SEED_JSON="$seed_json"
export CREDENTIALS_JSON="$credentials_json"

# Omnibus generates this tiny launcher file in the main container's image layer,
# which is intentionally not a shared volume. The seed container needs the same
# installation identity to run Rails against the shared configuration and data.
install -d -m 0755 /opt/gitlab/etc
if [[ ! -f /opt/gitlab/etc/gitlab-rails-rc ]]; then
  printf '%s\n' \
    "gitlab_user='git'" \
    "gitlab_group='git'" \
    "registry_dir=''" \
    "registry_user='registry'" \
    "registry_group='registry'" \
    > /opt/gitlab/etc/gitlab-rails-rc
fi

install -d -m 0755 /opt/gitlab/etc/gitlab-rails/env
declare -A rails_env=(
  [SSL_CERT_DIR]='/opt/gitlab/embedded/ssl/certs/'
  [SSL_CERT_FILE]='/opt/gitlab/embedded/ssl/cert.pem'
  [TZ]=':/etc/localtime'
  [EXECJS_RUNTIME]='Disabled'
  [HOME]='/var/opt/gitlab'
  [SIDEKIQ_MEMORY_KILLER_MAX_RSS]='2000000'
  [BUNDLE_GEMFILE]='/opt/gitlab/embedded/service/gitlab-rails/Gemfile'
  [PYTHONPATH]='/opt/gitlab/embedded/lib/python3.9/site-packages'
  [PUMA_WORKER_MAX_MEMORY]=''
  [ICU_DATA]='/opt/gitlab/embedded/share/icu/current'
  [PATH]='/opt/gitlab/bin:/opt/gitlab/embedded/bin:/bin:/usr/bin'
  [RAILS_ENV]='production'
)
for name in "${!rails_env[@]}"; do
  printf '%s' "${rails_env[$name]}" > "/opt/gitlab/etc/gitlab-rails/env/$name"
done

rails_root='/opt/gitlab/embedded/service/gitlab-rails'
for name in gitlab.yml secrets.yml cable.yml resque.yml redis.yml click_house.yml session_store.yml database.yml; do
  ln -sfn "/var/opt/gitlab/gitlab-rails/etc/$name" "$rails_root/config/$name"
done
ln -sfn /var/log/gitlab/gitlab-rails "$rails_root/log"
ln -sfn /var/opt/gitlab/gitlab-rails/tmp "$rails_root/tmp"
for name in gitlab_kas_secret gitlab_pages_secret gitlab_workhorse_secret gitlab_shell_secret; do
  ln -sfn "/var/opt/gitlab/gitlab-rails/etc/$name" "$rails_root/.$name"
done
ln -sfn "$rails_root/.gitlab_shell_secret" \
  /opt/gitlab/embedded/service/gitlab-shell/.gitlab_shell_secret

psql_as_owner() {
  chpst -u gitlab-psql /opt/gitlab/embedded/bin/psql \
    -h /var/opt/gitlab/postgresql \
    -d gitlabhq_production \
    -v ON_ERROR_STOP=1 \
    -c "$1"
}

revoke_import_privilege() {
  psql_as_owner 'ALTER ROLE gitlab NOSUPERUSER;' >/dev/null
}

psql_as_owner 'ALTER ROLE gitlab SUPERUSER;' >/dev/null
trap revoke_import_privilege EXIT
gitlab-rails runner /seed/import-gitlab.rb
gitlab-rails runner /seed/create-blank-projects.rb
revoke_import_privilege
trap - EXIT

printf '%s\n' "$seed_hash" > "$marker"
echo "GitLab JSON seeding and credential setup completed successfully."
