#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-$ROOT_DIR/user-credentials.json}"
OUTPUT_CREDENTIALS_FILE="${OUTPUT_CREDENTIALS_FILE:-$ROOT_DIR/generated-mcp-credentials.json}"
USER_NAME=""
APPS="rocketchat,owncloud,frappe,zammad"
usage() { echo "Usage: $0 --user USER [--apps csv]"; }
while (($#)); do
  case "$1" in
    --user) USER_NAME="${2:?missing username}"; shift 2;;
    --apps) APPS="${2:?missing app list}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done
[[ -n "$USER_NAME" ]] || { usage >&2; exit 2; }
command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
[[ -r "$CREDENTIALS_FILE" ]] || { echo "cannot read $CREDENTIALS_FILE" >&2; exit 1; }
cp "$CREDENTIALS_FILE" "$OUTPUT_CREDENTIALS_FILE"

readarray -t user_fields < <(python3 - "$CREDENTIALS_FILE" "$USER_NAME" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
u=next((x for x in d['users'] if x['username']==sys.argv[2]),None)
if not u: raise SystemExit(f'unknown user: {sys.argv[2]}')
print(u['password'])
print(u.get('email') or u['username'])
admin=next(x for x in d['users'] if x['username']=='admin')
print(admin['password'])
PY
)
password="${user_fields[0]}"
user_email="${user_fields[1]}"
admin_password="${user_fields[2]}"

set_credential() {
  local app="$1" key="$2" value="$3"
  python3 - "$OUTPUT_CREDENTIALS_FILE" "$USER_NAME" "$app" "$key" "$value" <<'PY'
import json,sys,tempfile,os
p,user,app,key,value=sys.argv[1:]
with open(p,encoding='utf-8') as f: d=json.load(f)
d.setdefault('app_credentials',{}).setdefault(user,{}).setdefault(app,{})[key]=value
fd,tmp=tempfile.mkstemp(dir=os.path.dirname(p),prefix='.credentials-'); os.close(fd)
with open(tmp,'w',encoding='utf-8') as f:
    json.dump(d,f,indent=2); f.write('\n')
os.chmod(tmp,os.stat(p).st_mode); os.replace(tmp,p)
PY
}

has_app() { [[ ",${APPS}," == *",$1,"* ]]; }

if has_app rocketchat; then
  base="${ROCKETCHAT_URL:-http://localhost:3000}"
  payload="$(python3 -c 'import json,sys; print(json.dumps({"user":sys.argv[1],"password":sys.argv[2]}))' "$USER_NAME" "$password")"
  response="$(curl -fsS --max-time 30 -H 'Content-Type: application/json' -d "$payload" "$base/api/v1/login")"
  rid="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["userId"])' <<<"$response")"
  rtoken="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["authToken"])' <<<"$response")"
  set_credential rocketchat rocketchat_user_id "$rid"
  set_credential rocketchat rocketchat_auth_token "$rtoken"
  echo "rocketchat: generated user id and auth token"
fi

if has_app owncloud; then
  set_credential owncloud owncloud_username "$USER_NAME"
  set_credential owncloud owncloud_password "$password"
  echo "owncloud: stored username/password (no token required)"
fi

if has_app frappe; then
  base="${FRAPPE_URL:-http://localhost:8084}"
  cookie_jar="$(mktemp)"
  curl -fsS --max-time 30 -c "$cookie_jar" -X POST \
    --data-urlencode 'usr=Administrator' --data-urlencode "pwd=$admin_password" \
    "$base/api/method/login" >/dev/null
  frappe_user="$user_email"; [[ "$USER_NAME" == admin ]] && frappe_user=Administrator
  response="$(curl -fsS --max-time 30 -b "$cookie_jar" -X POST \
    --data-urlencode "user=$frappe_user" \
    "$base/api/method/frappe.core.doctype.user.user.generate_keys")"
  rm -f "$cookie_jar"
  fkey="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["message"]["api_key"])' <<<"$response")"
  fsecret="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["message"]["api_secret"])' <<<"$response")"
  set_credential frappe frappe_api_key "$fkey"; set_credential frappe frappe_api_secret "$fsecret"
  echo "frappe: generated API key and secret"
fi

if has_app zammad; then
  base="${ZAMMAD_URL:-http://localhost:8050}"
  response="$(curl -fsS --max-time 30 -u "$USER_NAME:$password" \
    -H 'Content-Type: application/json' -X POST "$base/api/v1/user_access_token" \
    -d '{"name":"enterprise-arena-mcp","permission":["ticket.agent","ticket.customer","user_preferences.access_token"]}')"
  ztoken="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])' <<<"$response")"
  set_credential zammad zammad_token "$ztoken"
  echo "zammad: generated access token"
fi

echo "Generated credentials written to $OUTPUT_CREDENTIALS_FILE"
