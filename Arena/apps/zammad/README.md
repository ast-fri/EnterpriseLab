# Seeded Zammad

This stack imports `Extracted_data/zammad_from_db.json` into isolated Docker
volumes, preserves its tickets and related records, and activates the same 13
local users used by Rocket.Chat and Dolibarr.

Reset and seed from the repository root:

```bash
./zammad/seed.sh \
  Extracted_data/zammad_from_db.json \
  zammad/docker-compose.yml
```

The UI is available at <http://localhost:8050>. Credentials are in
`user-credentials.json`. The `admin` user is the sole
administrator; the other 12 users are agents with full access to all four
active ticket groups. Historical export-only users remain as inactive reference
records so imported tickets keep valid owners and customers.

After seeding, ordinary starts reuse the seeded database:

```bash
docker compose -p zammad-seeded \
  -f zammad/docker-compose.yml up -d
```

Run `seed.sh` again to remove only the `zammad-seeded` containers and volumes
and reconstruct the baseline from JSON. The built-in backup service is behind
the optional `backup` profile because this workflow treats JSON as the baseline.
