import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/code")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "plane.settings.production")

import django

django.setup()

from django.apps import apps
from django.db import connection, transaction
from psycopg.types.json import Jsonb


EXPORT_PATH = Path("/seed/plane_from_db.json")
CREDENTIALS_PATH = Path("/seed/user-credentials.json")


def model_for_table(table_name):
    return next(
        (model for model in apps.get_models() if model._meta.db_table == table_name),
        None,
    )


def concrete_field_names(model):
    return {field.name for field in model._meta.concrete_fields}


def supported_defaults(model, values):
    fields = concrete_field_names(model)
    return {key: value for key, value in values.items() if key in fields}


def restore_export(export):
    tables = export["tables"]
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
        available_tables = {row[0] for row in cursor.fetchall()}
        restore_tables = [name for name in tables if name in available_tables]
        missing = sorted(set(tables) - available_tables)
        if missing:
            print("plane-seed: ignoring export tables absent from this release", missing)

        quoted_tables = ", ".join(connection.ops.quote_name(name) for name in restore_tables)
        with transaction.atomic():
            cursor.execute("SET LOCAL session_replication_role = replica")
            cursor.execute(f"TRUNCATE TABLE {quoted_tables} RESTART IDENTITY CASCADE")

            for table_name in restore_tables:
                rows = tables[table_name]
                if not rows:
                    continue
                cursor.execute(
                    "SELECT column_name, udt_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "AND is_generated = 'NEVER' ORDER BY ordinal_position",
                    [table_name],
                )
                column_types = dict(cursor.fetchall())
                columns = [column for column in rows[0] if column in column_types]
                quoted_columns = ", ".join(connection.ops.quote_name(column) for column in columns)
                placeholders = ", ".join(["%s"] * len(columns))
                sql = (
                    f"INSERT INTO {connection.ops.quote_name(table_name)} "
                    f"({quoted_columns}) VALUES ({placeholders})"
                )
                values = []
                for row in rows:
                    record = []
                    for column in columns:
                        item = row.get(column)
                        if item is not None and column_types[column] in {"json", "jsonb"}:
                            item = Jsonb(item)
                        record.append(item)
                    values.append(record)
                cursor.executemany(sql, values)

    print(
        "plane-seed: restored export",
        {name: len(rows) for name, rows in tables.items() if rows},
    )


def create_shared_users(credentials):
    User = model_for_table("users")
    Profile = model_for_table("profiles")
    Workspace = model_for_table("workspaces")
    WorkspaceMember = model_for_table("workspace_members")
    Project = model_for_table("projects")
    ProjectMember = model_for_table("project_members")
    Instance = model_for_table("instances")
    InstanceAdmin = model_for_table("instance_admins")
    required = {
        "User": User,
        "Profile": Profile,
        "Workspace": Workspace,
        "WorkspaceMember": WorkspaceMember,
        "Project": Project,
        "ProjectMember": ProjectMember,
        "Instance": Instance,
        "InstanceAdmin": InstanceAdmin,
    }
    missing = [name for name, model in required.items() if model is None]
    if missing:
        raise RuntimeError(f"Plane models unavailable: {', '.join(missing)}")

    workspace = Workspace.objects.get(slug="inazuma-engineering")
    projects = list(Project.objects.filter(workspace=workspace, deleted_at__isnull=True))
    shared_users = []

    with transaction.atomic():
        for credential in credentials:
            username = credential["username"]
            email = credential["email"]
            first_name, _, last_name = username.partition(".")
            user = User.objects.filter(email__iexact=email).first()
            if user is None:
                user = User(email=email)
            user.username = username
            user.email = email
            user.first_name = credential.get("first_name", first_name.title())
            user.last_name = credential.get("last_name", last_name.title())
            user.display_name = username
            user.is_active = True
            user.is_email_verified = True
            user.is_password_expired = False
            user.is_password_autoset = False
            user.is_staff = credential.get("admin", False)
            user.is_superuser = credential.get("admin", False)
            user.set_password(credential["password"])
            user.save()
            shared_users.append(user)

            profile_defaults = supported_defaults(
                Profile,
                {
                    "is_onboarded": True,
                    "is_tour_completed": True,
                    "onboarding_step": {
                        "workspace_join": True,
                        "profile_complete": True,
                        "workspace_create": True,
                        "workspace_invite": True,
                    },
                    "use_case": "Engineering",
                    "role": "Individual contributor",
                    "last_workspace_id": workspace.id,
                    "billing_address_country": "INDIA",
                    "language": "en",
                },
            )
            Profile.objects.update_or_create(user=user, defaults=profile_defaults)

        admin_user = next(user for user in shared_users if user.is_superuser)
        owner = User.objects.filter(id=workspace.owner_id).first() or admin_user
        for user in shared_users:
            role = 20 if user in {admin_user, owner} else 15
            member_defaults = supported_defaults(
                WorkspaceMember,
                {
                    "role": role,
                    "is_active": True,
                    "created_by": owner,
                    "updated_by": owner,
                },
            )
            WorkspaceMember.objects.update_or_create(
                workspace=workspace,
                member=user,
                defaults=member_defaults,
            )
            for project in projects:
                project_defaults = supported_defaults(
                    ProjectMember,
                    {
                        "role": role,
                        "is_active": True,
                        "created_by": owner,
                        "updated_by": owner,
                    },
                )
                ProjectMember.objects.update_or_create(
                    workspace=workspace,
                    project=project,
                    member=user,
                    defaults=project_defaults,
                )

        instance = Instance.objects.first()
        if instance:
            admin_defaults = supported_defaults(
                InstanceAdmin,
                {"role": 20, "is_verified": True},
            )
            InstanceAdmin.objects.update_or_create(
                instance=instance,
                user=admin_user,
                defaults=admin_defaults,
            )

        for transient_table in ("sessions", "api_tokens"):
            transient_model = model_for_table(transient_table)
            if transient_model:
                transient_model.objects.all().delete()

    print(
        "plane-seed: shared users ready",
        {
            "users": User.objects.filter(is_active=True).count(),
            "workspace_members": WorkspaceMember.objects.filter(workspace=workspace).count(),
            "projects": len(projects),
            "project_members": ProjectMember.objects.filter(workspace=workspace).count(),
        },
    )


def main():
    export = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    credential_data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    if export.get("service") != "plane" or export.get("database") != "postgresql":
        raise RuntimeError("Expected a Plane PostgreSQL JSON export")
    restore_export(export)
    create_shared_users(credential_data["users"])


if __name__ == "__main__":
    main()
