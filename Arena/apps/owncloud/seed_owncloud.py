#!/usr/bin/env python3
"""Rebuild portable ownCloud state from the extracted database JSON."""

from __future__ import annotations

import argparse
import base64
import io
import json
import posixpath
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath


DIRECTORY_MIME = "httpd/unix-directory"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
JPEG_1X1 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EB//2Q=="
)


def basic_auth(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def request(url: str, method: str, username: str, password: str, data: bytes | None = None,
            headers: dict[str, str] | None = None, accepted: tuple[int, ...] = (200, 201, 204)) -> bytes:
    merged = {"Authorization": basic_auth(username, password), **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=merged, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read()
            if response.status not in accepted:
                raise RuntimeError(f"{method} {url}: HTTP {response.status}: {body[:500]!r}")
            return body
    except urllib.error.HTTPError as exc:
        body = exc.read()
        if exc.code not in accepted:
            raise RuntimeError(f"{method} {url}: HTTP {exc.code}: {body[:500]!r}") from exc
        return body


def ocs(base_url: str, admin: dict, method: str, path: str, fields: dict[str, str]) -> dict:
    data = urllib.parse.urlencode(fields).encode()
    body = request(
        f"{base_url}/ocs/v1.php/cloud/{path}?format=json",
        method,
        admin["username"],
        admin["password"],
        data=data,
        headers={
            "OCS-APIRequest": "true",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    parsed = json.loads(body)
    meta = parsed["ocs"]["meta"]
    if int(meta["statuscode"]) not in (100, 102):
        raise RuntimeError(f"OCS {method} {path}: {meta}")
    return parsed["ocs"]


def safe_export_path(raw: str) -> str | None:
    if raw == "files":
        return None
    if not raw.startswith("files/"):
        return None
    relative = raw[6:].strip("/")
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return str(path)


def extract_file_trees(export: dict, allowed_users: set[str]) -> dict[str, list[dict]]:
    tables = export.get("tables", {})
    storage_owner = {}
    for row in tables.get("oc_storages", []):
        storage_id = str(row.get("id", ""))
        if storage_id.startswith("home::"):
            storage_owner[str(row.get("numeric_id"))] = storage_id[6:]
    mimetypes = {str(row.get("id")): row.get("mimetype", "application/octet-stream")
                 for row in tables.get("oc_mimetypes", [])}
    result: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in tables.get("oc_filecache", []):
        owner = storage_owner.get(str(row.get("storage")))
        path = safe_export_path(str(row.get("path", "")))
        if owner not in allowed_users or path is None or (owner, path) in seen:
            continue
        seen.add((owner, path))
        result[owner].append({
            "path": path,
            "mime": mimetypes.get(str(row.get("mimetype")), "application/octet-stream"),
            "original_size": int(row.get("size", 0)),
            "mtime": str(row.get("mtime", "")),
        })
    return result


def minimal_pdf(owner: str, path: str) -> bytes:
    text = f"Dummy ownCloud file reconstructed for {owner}: {path}"
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(safe) + 34} >>\nstream\nBT /F1 12 Tf 50 740 Td ({safe}) Tj ET\nendstream".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = io.BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(output.tell())
        output.write(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return output.getvalue()


def minimal_odt(owner: str, path: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/manifest.xml", """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">
<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>
<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
</manifest:manifest>""")
        archive.writestr("content.xml", f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><text:p>Dummy ownCloud file reconstructed for {owner}: {path}</text:p></office:text></office:body></office:document-content>""")
    return output.getvalue()


def dummy_content(owner: str, item: dict) -> bytes:
    path, mime = item["path"], item["mime"]
    if mime == "application/pdf" or path.lower().endswith(".pdf"):
        return minimal_pdf(owner, path)
    if mime == "image/png" or path.lower().endswith(".png"):
        return PNG_1X1
    if mime == "image/jpeg" or path.lower().endswith((".jpg", ".jpeg")):
        return JPEG_1X1
    if mime == "application/vnd.oasis.opendocument.text" or path.lower().endswith(".odt"):
        return minimal_odt(owner, path)
    prefix = "# " if path.lower().endswith(".py") else ""
    return (
        f"{prefix}Placeholder reconstructed from Extracted_data/owncloud_from_db.json\n"
        f"{prefix}Owner: {owner}\n{prefix}Original path: {path}\n"
        f"{prefix}Original MIME type: {mime}\n{prefix}Original size: {item['original_size']} bytes\n"
    ).encode()


def dav_url(base_url: str, username: str, relative: str = "") -> str:
    components = [urllib.parse.quote(username, safe="")]
    components.extend(urllib.parse.quote(part, safe="") for part in PurePosixPath(relative).parts)
    return f"{base_url}/remote.php/dav/files/" + "/".join(components)


def seed_files(base_url: str, user: dict, entries: list[dict]) -> tuple[int, int]:
    username, password = user["username"], user["password"]
    directories = set()
    files = []
    for item in entries:
        path = item["path"]
        if item["mime"] == DIRECTORY_MIME:
            directories.add(path)
        else:
            files.append(item)
            parent = posixpath.dirname(path)
            while parent:
                directories.add(parent)
                parent = posixpath.dirname(parent)
    for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
        request(dav_url(base_url, username, directory), "MKCOL", username, password, accepted=(201, 405))
    for item in sorted(files, key=lambda value: value["path"]):
        request(
            dav_url(base_url, username, item["path"]),
            "PUT",
            username,
            password,
            data=dummy_content(username, item),
            headers={"Content-Type": item["mime"]},
            accepted=(201, 204),
        )
    return len(directories), len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    with open(args.json, encoding="utf-8") as handle:
        export = json.load(handle)
    with open(args.credentials, encoding="utf-8") as handle:
        users = json.load(handle)["users"]
    admin = next(user for user in users if user.get("admin"))

    ocs(base_url, admin, "POST", "groups", {"groupid": "Employees"})
    for user in users:
        if not user.get("admin"):
            ocs(base_url, admin, "POST", "users", {
                "userid": user["username"],
                "password": user["password"],
            })
        quoted = urllib.parse.quote(user["username"], safe="")
        for key, value in (("display", user["display_name"]), ("email", user["email"]), ("password", user["password"])):
            ocs(base_url, admin, "PUT", f"users/{quoted}", {"key": key, "value": value})
        if not user.get("admin"):
            ocs(base_url, admin, "POST", f"users/{quoted}/groups", {"groupid": "Employees"})

    trees = extract_file_trees(export, {user["username"] for user in users})
    total_directories = total_files = 0
    for user in users:
        entries = trees.get(user["username"], [])
        directories, files = seed_files(base_url, user, entries)
        total_directories += directories
        total_files += files
        print(f"seeded {user['username']}: {directories} directories, {files} dummy files")
    print(f"seed complete: {len(users)} users, {total_directories} directories, {total_files} dummy files")


if __name__ == "__main__":
    main()
