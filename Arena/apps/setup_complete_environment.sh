#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
MCP_DIR="${MCP_DIR:-$ROOT_DIR/../MCP_servers}"
USER_NAME="${MCP_USER:-surya.reddy}"
MODE=continue
while (($#)); do
  case "$1" in
    --user) USER_NAME="${2:?missing username}"; shift 2;;
    --reset) MODE=reset; shift;;
    --continue) MODE=continue; shift;;
    -h|--help) echo "Usage: $0 [--user USER] [--reset|--continue]"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ -d "$MCP_DIR" ]] || { echo "MCP directory not found: $MCP_DIR" >&2; exit 1; }
"$ROOT_DIR/start_all_servers.sh" "--$MODE"
"$ROOT_DIR/generate_app_tokens.sh" --user "$USER_NAME" --apps rocketchat,owncloud,frappe,zammad
get(){ python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["app_credentials"][sys.argv[2]][sys.argv[3]][sys.argv[4]])' "$ROOT_DIR/generated-mcp-credentials.json" "$USER_NAME" "$1" "$2"; }
setenv(){ local f="$1" k="$2" v="$3"; sed -i -E "s#^([[:space:]]*-[[:space:]]*$k=).*#\\1$v#" "$MCP_DIR/$f"; }
setenv gitlab/docker-compose.yml GITLAB_PERSONAL_ACCESS_TOKEN "$(get gitlab gitlab_token)"
setenv rocketchat/docker-compose.yml ROCKETCHAT_USER_ID "$(get rocketchat rocketchat_user_id)"
setenv rocketchat/docker-compose.yml ROCKETCHAT_AUTH_TOKEN "$(get rocketchat rocketchat_auth_token)"
setenv dolibarr/docker-compose.yml DOLIBARR_API_KEY "$(get dolibarr dolibarr_api_key)"
setenv owncloud/docker-compose.yml OWNCLOUD_USERNAME "$(get owncloud owncloud_username)"
setenv owncloud/docker-compose.yml OWNCLOUD_PASSWORD "$(get owncloud owncloud_password)"
setenv plane/docker-compose.yml PLANE_API_KEY "$(get plane plane_api_key)"
setenv plane/docker-compose.yml PLANE_WORKSPACE_SLUG "$(get plane plane_workspace_slug)"
(cd "$MCP_DIR" && ./start_all_servers.sh)
