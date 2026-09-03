# Seeded Enterprise Applications

This directory contains reproducible Docker environments for Rocket.Chat,
GitLab, Dolibarr, Zammad, Frappe HRMS, Plane, and ownCloud. The sanitized JSON
baselines are under `Extracted_data/`, and all applications use the same 13
dummy identities from `user-credentials.json`.

No upstream Frappe or Zammad Git repository needs to be cloned. Their Compose
files use official container images and mount the seed overlay committed here.

## First-time setup

From `Arena/apps`:

```bash
./start_all_servers.sh --reset
```

This creates fresh isolated volumes and seeds every application. The first run
can take more than an hour because large images and application dependencies
must be downloaded.

## Reuse or stop the current state

```bash
./start_all_servers.sh --continue
./start_all_servers.sh --stop
./start_all_servers.sh --status
```

## Start applications and MCP servers

After the applications have been seeded at least once:

```bash
./setup_complete_environment.sh --user surya.reddy --continue
```

Use `--reset` instead of `--continue` to rebuild all application baselines.
The setup script refreshes user-specific API credentials, writes them into the
Compose definitions under `../MCP_servers`, and starts those MCP servers last.

The credentials are deliberately dummy evaluation credentials. Never reuse
their passwords or generated tokens outside this disposable environment.
