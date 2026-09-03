# Seeded Dolibarr

This stack creates an isolated Dolibarr 23.0.2 instance from
`Extracted_data/dolibarr_from_db.json`. It uses named Docker volumes and never
uses the older host-mounted Dolibarr database or document directories.

The first start on empty volumes imports all exported tables before the web
health check succeeds. It also activates the 13 Rocket.Chat seed users and
sets the same usernames and passwords from `user-credentials.json`. Only these 13
accounts are visible in Users & Groups; other exported users remain hidden
reference records so historical author IDs are preserved. Non-admin seed users
receive all enabled business-module rights, excluding user/group administration.

## Reset to the JSON baseline

From the repository root:

```bash
./dolibarr/seed.sh \
  Extracted_data/dolibarr_from_db.json \
  dolibarr/docker-compose.yml
```

This deletes only the `dolibarr-seeded` containers and named volumes, then
starts and verifies a fresh import.

## Normal restart

```bash
docker compose -p dolibarr-seeded -f dolibarr/docker-compose.yml up -d
```

A normal restart retains the seeded database. The UI is available at
http://localhost:8082.

To use another host port, set `DOLIBARR_HOST_PORT` for both seeding and later
Compose commands.
