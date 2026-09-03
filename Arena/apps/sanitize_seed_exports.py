#!/usr/bin/env python3
"""Clean extracted seed exports for publishable re-imports."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "Extracted_data"

FIXED_USERS = {
    "admin",
    "abigail.mitchell",
    "aarav.mittal",
    "surya.reddy",
    "raj.patel",
    "rahul.khanna",
    "karan.sharma",
    "priya.arora",
    "sameer.malhotra",
    "ethan.reynolds",
    "anjali.mathew",
    "vandana.reddy",
    "neeraj.sharma",
}

CANONICAL_NAMES = {
    "admin": "Administrator",
    "abigail.mitchell": "Abigail Mitchell",
    "aarav.mittal": "Aarav Mittal",
    "surya.reddy": "Surya Reddy",
    "raj.patel": "Raj Patel",
    "rahul.khanna": "Rahul Khanna",
    "karan.sharma": "Karan Sharma",
    "priya.arora": "Priya Arora",
    "sameer.malhotra": "Sameer Malhotra",
    "ethan.reynolds": "Ethan Reynolds",
    "anjali.mathew": "Anjali Mathew",
    "vandana.reddy": "Vandana Reddy",
    "neeraj.sharma": "Neeraj Sharma",
}

SYSTEM_USERS = {
    "rocketchat": {"rocket.cat"},
    "frappe": {"guest"},
}

APPS = ("rocketchat", "gitlab", "dolibarr", "zammad", "frappe", "plane", "owncloud")


def normalized_username(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "@" in text:
        text = text.split("@", 1)[0]
    return text.replace(" ", ".")


def canonical_full_name(username: str) -> str:
    return CANONICAL_NAMES.get(username, username.replace(".", " ").title())


def canonical_email(username: str, domain: str = "inazuma.com") -> str:
    return f"{username}@{domain}"


def load_export(app: str) -> dict:
    path = EXPORT_DIR / f"{app}_from_db.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write(app: str, export: dict) -> None:
    path = EXPORT_DIR / f"{app}_from_db.json"
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def bucket(export: dict) -> dict:
    return export.get("tables") or export.get("collections") or {}


def update_statistics(export: dict) -> None:
    data = bucket(export)
    statistics = export.setdefault("statistics", {})
    key = "total_tables" if "tables" in export else "total_collections"
    populated_key = "tables_with_data" if "tables" in export else "collections_with_data"
    statistics[key] = len(data)
    statistics[populated_key] = sum(bool(rows) for rows in data.values())


def clear_tables(tables: dict, names: set[str]) -> int:
    cleared = 0
    for name in names:
        if name in tables:
            cleared += len(tables[name])
            tables[name] = []
    return cleared


def blank_fields(row: dict[str, Any], field_names: set[str]) -> None:
    for field in field_names:
        if field in row:
            row[field] = None


def blank_matching_fields(row: dict[str, Any], patterns: tuple[re.Pattern[str], ...]) -> None:
    for key in list(row):
        if any(pattern.search(key) for pattern in patterns):
            row[key] = None


def sanitize_dolibarr(export: dict) -> dict[str, int]:
    tables = export["tables"]
    users = tables["llx_user"]
    removed = [row for row in users if normalized_username(row.get("login")) not in FIXED_USERS]
    removed_ids = {str(row["rowid"]) for row in removed}
    tables["llx_user"] = [row for row in users if str(row["rowid"]) not in removed_ids]

    dropped_relations = 0
    for table_name in ("llx_user_rights", "llx_oauth_token"):
        rows = tables.get(table_name, [])
        kept = [row for row in rows if str(row.get("fk_user")) not in removed_ids]
        dropped_relations += len(rows) - len(kept)
        tables[table_name] = kept

    remapped_references = 0
    for row in tables.get("llx_project_task_timesheet", []):
        for field in ("fk_userid", "fk_user_modification"):
            if str(row.get(field)) in removed_ids:
                row[field] = "1"
                remapped_references += 1

    for row in tables["llx_user"]:
        username = normalized_username(row.get("login"))
        if username in FIXED_USERS:
            full_name = canonical_full_name(username)
            first_name, _, last_name = full_name.partition(" ")
            row["firstname"] = first_name
            row["lastname"] = last_name
            row["email"] = canonical_email(username)
            row["personal_email"] = canonical_email(username)
        blank_fields(
            row,
            {
                "api_key",
                "pass_encoding",
                "pass",
                "pass_temp",
                "birth",
                "birth_place",
                "office_phone",
                "user_mobile",
                "personal_mobile",
                "email_oauth2",
                "datelastlogin",
                "datepreviouslogin",
                "datelastpassvalidation",
                "flagdelsessionsbefore",
                "iplastlogin",
                "ippreviouslogin",
                "import_key",
            },
        )

    update_statistics(export)
    return {
        "users_removed": len(removed),
        "relations_removed": dropped_relations,
        "references_remapped": remapped_references,
    }


def replace_strings(value: Any, replacements: dict[str, str], id_map: dict[int, int]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
        for old, new in id_map.items():
            value = re.sub(rf"(?<=user_id:)\s*{old}\b", f" {new}", value)
            value = re.sub(rf"(?<=customer_id:)\s*{old}\b", str(new), value)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements, id_map) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements, id_map) for key, item in value.items()}
    return value


def sanitize_zammad(export: dict) -> dict[str, int]:
    tables = export["tables"]
    users = tables["users"]
    by_username = {normalized_username(row.get("login") or row.get("email")): row for row in users}
    replacements_by_username = {
        "anita.menon": "surya.reddy",
        "prakash.iyer": "sameer.malhotra",
        "finance.ops": "priya.arora",
    }
    id_map: dict[int, int] = {}
    replacements: dict[str, str] = {}
    for old_username, new_username in replacements_by_username.items():
        old = by_username.get(old_username)
        new = by_username.get(new_username)
        if old is None:
            continue
        if new is None:
            raise RuntimeError(f"Zammad replacement user is absent: {new_username}")
        id_map[int(old["id"])] = int(new["id"])
        for field in ("login", "email"):
            if old.get(field) and new.get(field):
                replacements[str(old[field])] = str(new[field])
        old_name = " ".join(filter(None, (old.get("firstname"), old.get("lastname"))))
        new_name = " ".join(filter(None, (new.get("firstname"), new.get("lastname"))))
        if old_name and new_name:
            replacements[old_name] = new_name

    extra_ids = set(id_map)
    tables["users"] = [row for row in users if int(row["id"]) not in extra_ids]

    avatars = tables.get("avatars", [])
    tables["avatars"] = [
        row for row in avatars
        if not (int(row.get("object_lookup_id") or 0) == 3 and int(row.get("o_id") or 0) in extra_ids)
    ]
    avatars_removed = len(avatars) - len(tables["avatars"])

    user_reference = re.compile(r"(?:^user_id$|^customer_id$|^owner_id$|_by_id$)")
    remapped_references = 0
    for table_name, rows in tables.items():
        for row in rows:
            for field, value in list(row.items()):
                if user_reference.search(field) and isinstance(value, (int, str)):
                    try:
                        numeric = int(value)
                    except (TypeError, ValueError):
                        continue
                    if numeric in id_map:
                        row[field] = str(id_map[numeric]) if isinstance(value, str) else id_map[numeric]
                        remapped_references += 1
            sanitized = replace_strings(row, replacements, id_map)
            row.clear()
            row.update(sanitized)

    for row in tables.get("users", []):
        username = normalized_username(row.get("login") or row.get("email"))
        if username in FIXED_USERS:
            full_name = canonical_full_name(username)
            first_name, _, last_name = full_name.partition(" ")
            row["firstname"] = first_name
            row["lastname"] = last_name
            row["name"] = full_name
            row["email"] = canonical_email(username)
            row["login"] = username
        blank_matching_fields(
            row,
            (
                re.compile(r"(?:^|_)(?:password|token|secret|session|cookie|ip|otp|public_email|incoming_email|feed_token|static_object_token|unlock_token|reset_password|avatar|phone|fax|mobile|street|zip|city|country|address)$", re.IGNORECASE),
            ),
        )

    for row in tables.get("settings", []):
        setting_id = str(row.get("name") or "")
        if setting_id == "application_secret":
            stable_secret = "enterprise-arena-public-seed-secret-v1"
            row["state_current"] = f"---\nvalue: {stable_secret}\n"
            row["state_initial"] = f"---\nvalue: {stable_secret}\n"
            continue
        if setting_id.startswith("Cloud_") or setting_id.startswith("Register_Server"):
            row["state_current"] = "---\nvalue: ''\n"
            row["state_initial"] = "---\nvalue: ''\n"
            continue
        if re.search(r"(secret|token|password|private|publickey|federation|matrix|client_secret|oauth)", setting_id, re.IGNORECASE):
            row["state_current"] = "---\nvalue: ''\n"
            row["state_initial"] = "---\nvalue: ''\n"

    clear_tables(
        tables,
        {
            "tokens",
            "user_devices",
            "online_notifications",
            "taskbars",
            "stats_stores",
            "sessions",
            "chats",
            "chat_sessions",
            "chat_messages",
            "avatars",
            "translations",
            "http_logs",
            "delayed_jobs",
            "active_job_locks",
            "jobs",
            "import_jobs",
            "external_credentials",
            "external_syncs",
            "failed_emails",
            "activity_streams",
            "cti_logs",
            "cti_caller_ids",
        },
    )

    update_statistics(export)
    return {
        "users_removed": len(extra_ids),
        "avatars_removed": avatars_removed,
        "references_remapped": remapped_references,
    }


def build_gitlab_routes(namespaces: list[dict], projects: list[dict]) -> list[dict]:
    namespace_by_id = {int(row["id"]): row for row in namespaces if row.get("id") is not None}

    def full_path(namespace_id: int) -> str:
        row = namespace_by_id[namespace_id]
        path = str(row.get("path") or "")
        parent_id = row.get("parent_id")
        if parent_id in (None, "", 0):
            return path
        parent_id = int(parent_id)
        parent = namespace_by_id.get(parent_id)
        if not parent:
            return path
        parent_path = full_path(parent_id)
        return f"{parent_path}/{path}" if path else parent_path

    routes = []
    next_id = 1
    for namespace in sorted(namespace_by_id.values(), key=lambda row: int(row["id"])):
        namespace_id = int(namespace["id"])
        routes.append({
            "id": next_id,
            "source_id": namespace_id,
            "source_type": "Namespace",
            "path": full_path(namespace_id),
            "created_at": namespace.get("created_at"),
            "updated_at": namespace.get("updated_at"),
            "name": namespace.get("name"),
            "namespace_id": namespace_id,
        })
        next_id += 1

    return routes


def build_gitlab_namespace_details(namespaces: list[dict], projects: list[dict]) -> list[dict]:
    rows = []
    for namespace in sorted(namespaces, key=lambda row: int(row["id"])):
        namespace_id = int(namespace["id"])
        rows.append({
            "namespace_id": namespace_id,
            "created_at": namespace.get("created_at"),
            "updated_at": namespace.get("updated_at"),
            "cached_markdown_version": 0,
            "description": namespace.get("description") or "",
            "description_html": "",
            "creator_id": namespace.get("owner_id"),
            "deleted_at": None,
            "state_metadata": None,
        })
    for project in sorted(projects, key=lambda row: int(row["id"])):
        namespace_id = int(project["project_namespace_id"])
        rows.append({
            "namespace_id": namespace_id,
            "created_at": project.get("created_at"),
            "updated_at": project.get("updated_at"),
            "cached_markdown_version": 0,
            "description": project.get("description") or "",
            "description_html": "",
            "creator_id": project.get("creator_id"),
            "deleted_at": None,
            "state_metadata": None,
        })
    return rows


def sanitize_gitlab(export: dict) -> dict[str, int]:
    tables = export["tables"]
    users = tables.get("users", [])
    for row in users:
        username = normalized_username(row.get("username") or row.get("email"))
        if username in FIXED_USERS:
            full_name = canonical_full_name(username)
            first_name, _, last_name = full_name.partition(" ")
            row["name"] = full_name
            row["username"] = username
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["email"] = canonical_email(username)
            row["notification_email"] = canonical_email(username)
            row["commit_email"] = canonical_email(username)
        row["encrypted_password"] = ""
        row["reset_password_token"] = None
        row["reset_password_sent_at"] = None
        row["remember_created_at"] = None
        row["current_sign_in_at"] = None
        row["last_sign_in_at"] = None
        row["current_sign_in_ip"] = None
        row["last_sign_in_ip"] = None
        row["password_expires_at"] = None
        row["avatar"] = None
        row["confirmation_token"] = None
        row["unconfirmed_email"] = None
        row["admin_email_unsubscribed_at"] = None
        row["encrypted_otp_secret"] = None
        row["encrypted_otp_secret_iv"] = None
        row["encrypted_otp_secret_salt"] = None
        row["otp_required_for_login"] = False
        row["public_email"] = None
        row["unlock_token"] = None
        row["incoming_email_token"] = None
        row["require_two_factor_authentication_from_group"] = None
        row["feed_token"] = None
        row["static_object_token"] = None
        row["static_object_token_encrypted"] = None
        row["otp_secret_expires_at"] = None

    projects = tables.get("projects", [])
    for row in projects:
        row["runners_token"] = None
        row["runners_token_encrypted"] = None
        row["external_webhook_token"] = None
        row["import_url"] = None
        row["import_type"] = None
        row["import_source"] = None
        row["avatar"] = None

    namespaces = tables.get("namespaces", [])
    project_namespaces = []
    for project in projects:
        synthetic_id = 10_000 + int(project["id"])
        parent = next(row for row in namespaces if int(row["id"]) == int(project["namespace_id"]))
        project_namespace = parent.copy()
        project_namespace.update({
            "id": synthetic_id,
            "name": project.get("name"),
            "path": project.get("path"),
            "owner_id": None,
            "type": "Project",
            "parent_id": int(project["namespace_id"]),
            "visibility_level": project.get("visibility_level"),
            "traversal_ids": list(parent.get("traversal_ids") or []) + [synthetic_id],
        })
        project_namespaces.append(project_namespace)
        project["project_namespace_id"] = synthetic_id
    tables["namespaces"] = namespaces + project_namespaces
    tables["routes"] = build_gitlab_routes(tables["namespaces"], projects)
    tables["namespace_details"] = build_gitlab_namespace_details(tables["namespaces"], projects)

    clear_tables(
        tables,
        {
            "audit_events",
            "authentication_events",
            "user_audit_events",
            "personal_access_tokens",
            "personal_access_token_last_used_ips",
            "emails",
            "issue_search_data",
            "merge_request_diff_commits",
            "merge_request_diff_commit_users",
        },
    )

    update_statistics(export)
    return {
        "users": len(users),
        "projects": len(projects),
        "namespaces": len(tables["namespaces"]),
        "routes": len(tables["routes"]),
    }


def sanitize_frappe(export: dict) -> dict[str, int]:
    tables = export["tables"]
    keep_tables = {
        "tabUser",
        "tabEmployee",
        "tabAttendance",
        "tabCustomer",
        "tabSupplier",
        "tabItem",
        "tabProject",
        "tabProject Template",
        "tabProject Template Task",
        "tabProject Type",
        "tabTask",
        "tabTask Depends On",
        "tabTask Type",
        "tabIssue",
        "tabIssue Type",
        "tabIssue Priority",
        "tabTimesheet",
        "tabTimesheet Detail",
        "tabToDo",
        "tabSales Invoice",
        "tabSales Invoice Item",
        "tabSales Invoice Payment",
        "tabSales Invoice Advance",
        "tabSales Order",
        "tabSales Order Item",
        "tabPurchase Order",
        "tabPurchase Order Item",
        "tabPurchase Order Item Supplied",
        "tabPayment Entry",
        "tabPayment Entry Deduction",
        "tabPayment Entry Reference",
        "tabJournal Entry",
        "tabJournal Entry Account",
        "tabJournal Entry Template",
        "tabJournal Entry Template Account",
    }
    tables_to_clear = [name for name in tables if name not in keep_tables]
    cleared_rows = clear_tables(tables, set(tables_to_clear))

    for row in tables.get("tabUser", []):
        username = normalized_username(row.get("username") or row.get("name") or row.get("email"))
        if username in FIXED_USERS:
            full_name = canonical_full_name(username)
            first_name, _, last_name = full_name.partition(" ")
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["full_name"] = full_name
            row["email"] = canonical_email(username)
            row["username"] = username
        blank_matching_fields(
            row,
            (
                re.compile(r"(?:^|_)(?:password|secret|token|session|cookie|ip|phone|mobile|fax|address|birth|dob|pan|bank|passport|emergency|image|email_signature|api_key|api_secret|reset_password|logout_all_sessions|last_password_reset_date|last_reset_password_key_generated_on)$", re.IGNORECASE),
            ),
        )

    for row in tables.get("tabEmployee", []):
        username = normalized_username(row.get("user_id") or row.get("personal_email") or row.get("company_email"))
        if username in FIXED_USERS:
            full_name = canonical_full_name(username)
            first_name, _, last_name = full_name.partition(" ")
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["employee_name"] = full_name
            row["personal_email"] = canonical_email(username)
            row["company_email"] = canonical_email(username)
            row["prefered_contact_email"] = canonical_email(username)
            row["prefered_email"] = canonical_email(username)
        if row.get("date_of_birth") not in (None, "", "NULL"):
            row["date_of_birth"] = "1990-01-01"
        blank_matching_fields(
            row,
            (
                re.compile(r"(?:^|_)(?:image|phone|mobile|address|passport|bank|emergency|attendance_device_id|current_address|permanent_address)$", re.IGNORECASE),
            ),
        )

    for table_name in (
        "tabUser Email",
        "tabUser Invitation",
        "tabUser Permission",
        "tabUser Role",
        "tabUser Type",
        "tabDocField",
        "tabOAuth Bearer Token",
        "tabOAuth Client",
        "tabOAuth Token",
        "tabOAuth Authorization Code",
        "tabUser Social Login",
        "tabSession Default",
        "tabCommunication",
        "tabEmail Queue",
        "tabEmail Queue Recipient",
        "tabError Log",
        "tabActivity Log",
        "tabVersion",
        "tabFile",
        "tabView Log",
        "tabHas Role",
        "tabDefaultValue",
        "tabCustom Field",
        "tabAddress",
        "tabAddress Template",
        "tabContact",
        "tabContact Email",
        "tabContact Phone",
        "tabLead",
        "tabLead Source",
        "tabProspect Lead",
        "tabOpportunity",
        "tabOpportunity Item",
        "tabOpportunity Lost Reason",
        "tabOpportunity Lost Reason Detail",
        "tabOpportunity Type",
        "tabProspect Opportunity",
        "tabGoogle Contacts",
        "tabSecurity Settings Contact",
    ):
        if table_name in tables:
            cleared_rows += len(tables[table_name])
            tables[table_name] = []

    update_statistics(export)
    return {"tables_cleared": len(tables_to_clear), "rows_cleared": cleared_rows}


def sanitize_plane(export: dict) -> dict[str, int]:
    tables = export["tables"]
    cleared_rows = clear_tables(
        tables,
        {
            "api_tokens",
            "sessions",
            "device_sessions",
            "api_activity_logs",
            "changelogs",
            "devices",
            "device_auth_tokens",
            "file_assets",
            "github_comment_syncs",
            "github_issue_syncs",
            "github_repositories",
            "github_repository_syncs",
            "importers",
            "instance_admins",
            "workspace_invites",
            "workspace_invitations",
        },
    )
    for row in tables.get("users", []):
        username = normalized_username(row.get("username") or row.get("email"))
        if username in FIXED_USERS:
            full_name = canonical_full_name(username)
            first_name, _, last_name = full_name.partition(" ")
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["display_name"] = full_name
            row["email"] = canonical_email(username)
            row["username"] = username
        row["password"] = "!"
        row["token"] = None
        row["token_updated_at"] = None
        blank_matching_fields(
            row,
            (
                re.compile(r"(?:^|_)(?:last_login|last_login_ip|last_logout_ip|last_login_medium|last_login_uagent|avatar|cover_image|location|mobile|session|ip|token|password)$", re.IGNORECASE),
            ),
        )
    update_statistics(export)
    return {"rows_cleared": cleared_rows}


def sanitize_owncloud(export: dict) -> dict[str, int]:
    tables = export["tables"]
    keep = {"oc_storages", "oc_mimetypes", "oc_filecache"}
    cleared_rows = 0
    for name in list(tables):
        if name not in keep:
            cleared_rows += len(tables[name])
            tables[name] = []
    return {"rows_cleared": cleared_rows}


def sanitize_rocketchat(export: dict) -> dict[str, int]:
    tables = export["collections"]
    cleared_rows = clear_tables(
        tables,
        {
            "rocketchat_import_data",
            "rocketchat_import",
            "rocketchat_nps",
            "rocketchat_statistics",
            "rocketchat_analytics",
            "rocketchat_server_events",
            "rocketchat_sessions",
            "rocketchat_workspace_credentials",
            "rocketchat_cron",
            "rocketchat_cron_history",
            "usersSessions",
            "rocketchat_federation_keys",
        },
    )
    for row in tables.get("rocketchat_settings", []):
        setting_id = str(row.get("_id") or "")
        if setting_id == "Organization_Email":
            row["value"] = "admin@rocketchat.local"
        if setting_id.startswith("Cloud_") or setting_id.startswith("Register_Server"):
            row["value"] = ""
            continue
        if re.search(r"(secret|token|password|private|publickey|federation|matrix|client_secret|oauth)", setting_id, re.IGNORECASE):
            for field in ("value", "packageValue", "envValue"):
                if field in row:
                    row[field] = ""
    for row in tables.get("rocketchat_avatars", []):
        row["token"] = None
    update_statistics(export)
    return {"rows_cleared": cleared_rows}


def account_usernames(app: str, export: dict) -> set[str]:
    data = bucket(export)
    if app == "rocketchat":
        return {normalized_username(row.get("username")) for row in data.get("users", [])}
    if app == "gitlab":
        return {normalized_username(row.get("username")) for row in data.get("users", [])}
    if app == "dolibarr":
        return {normalized_username(row.get("login")) for row in data.get("llx_user", [])}
    if app == "zammad":
        return {normalized_username(row.get("login") or row.get("email")) for row in data.get("users", [])}
    if app == "frappe":
        return {normalized_username(row.get("username") or row.get("name")) for row in data.get("tabUser", [])}
    if app == "plane":
        return {normalized_username(row.get("email") or row.get("username")) for row in data.get("users", [])}
    if app == "owncloud":
        return {normalized_username(row.get("uid")) for row in data.get("oc_users", [])}
    raise ValueError(app)


def audit_one(app: str, export: dict) -> bool:
    data = bucket(export)
    users = account_usernames(app, export)
    allowed = FIXED_USERS | SYSTEM_USERS.get(app, set())
    extras = sorted(users - allowed)
    fixed_count = len(users & FIXED_USERS)
    system_count = len(users & SYSTEM_USERS.get(app, set()))
    ok = not extras

    if app == "rocketchat":
        ok = ok and all(not data.get(name) for name in (
            "rocketchat_import_data",
            "rocketchat_import",
            "rocketchat_nps",
            "rocketchat_statistics",
            "rocketchat_analytics",
            "rocketchat_server_events",
            "rocketchat_sessions",
            "rocketchat_workspace_credentials",
            "rocketchat_cron",
            "rocketchat_cron_history",
            "usersSessions",
            "rocketchat_federation_keys",
        ))
    if app == "gitlab":
        ok = ok and all(not data.get(name) for name in (
            "audit_events",
            "authentication_events",
            "user_audit_events",
            "personal_access_tokens",
            "personal_access_token_last_used_ips",
            "emails",
            "issue_search_data",
            "merge_request_diff_commits",
            "merge_request_diff_commit_users",
        ))
        ok = ok and len(data.get("routes", [])) == 13
    if app == "zammad":
        ok = ok and all(not data.get(name) for name in ("tokens", "user_devices", "online_notifications", "taskbars", "stats_stores", "sessions", "chats", "chat_sessions", "chat_messages", "avatars", "translations"))
    if app == "owncloud":
        ok = ok and all(not data.get(name) for name in data if name not in {"oc_storages", "oc_mimetypes", "oc_filecache"})
    if app == "plane":
        ok = ok and not data.get("api_tokens") and not data.get("sessions")
    if app == "frappe":
        ok = ok and all(not data.get(name) for name in (
            "tabUser Email",
            "tabUser Invitation",
            "tabUser Permission",
            "tabUser Role",
            "tabUser Type",
            "tabDocField",
            "tabOAuth Bearer Token",
            "tabOAuth Client",
            "tabOAuth Token",
            "tabOAuth Authorization Code",
            "tabUser Social Login",
            "tabSession Default",
            "tabCommunication",
            "tabEmail Queue",
            "tabEmail Queue Recipient",
            "tabError Log",
            "tabActivity Log",
            "tabVersion",
            "tabFile",
            "tabView Log",
            "tabHas Role",
            "tabDefaultValue",
            "tabCustom Field",
            "tabAddress",
            "tabAddress Template",
            "tabContact",
            "tabContact Email",
            "tabContact Phone",
            "tabLead",
            "tabLead Source",
            "tabProspect Lead",
            "tabOpportunity",
            "tabOpportunity Item",
            "tabOpportunity Lost Reason",
            "tabOpportunity Lost Reason Detail",
            "tabOpportunity Type",
            "tabProspect Opportunity",
            "tabGoogle Contacts",
            "tabSecurity Settings Contact",
        ))

    print(f"{app}: fixed={fixed_count}, required_system={system_count}, extras={extras or 'none'}")
    return ok


def audit_all() -> bool:
    clean = True
    for app in APPS:
        clean = audit_one(app, load_export(app)) and clean
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if any exported account is outside the allowlist")
    mode.add_argument("--write", action="store_true", help="sanitize affected exports in place, then verify all exports")
    args = parser.parse_args()

    if args.write:
        sanitizers = {
            "rocketchat": sanitize_rocketchat,
            "gitlab": sanitize_gitlab,
            "dolibarr": sanitize_dolibarr,
            "zammad": sanitize_zammad,
            "frappe": sanitize_frappe,
            "plane": sanitize_plane,
            "owncloud": sanitize_owncloud,
        }
        for app, sanitizer in sanitizers.items():
            export = load_export(app)
            result = sanitizer(export)
            atomic_write(app, export)
            print(f"sanitized {app}: {result}")

    if not audit_all():
        raise SystemExit("Export account audit failed")


if __name__ == "__main__":
    main()
