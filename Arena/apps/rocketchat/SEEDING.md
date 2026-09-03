# Rocket.Chat baseline seeding

The Compose file uses the fixed project name `rocketchat-seeded`, starts NATS,
waits for MongoDB, and runs an idempotent one-shot seed service before starting
Rocket.Chat. A second one-shot service applies the preset passwords from
`user-credentials.json` after seeding and before Rocket.Chat starts.

From the `servers` directory, run:

```sh
./rocket-chat/seed.sh \
  Extracted_data/rocketchat_from_db.json \
  rocket-chat/docker-compose.yml
```

When the command succeeds, open `http://localhost:3000` and sign in with:

- Username: `admin`
- Password: `Admin123!`

Running the command again removes only the `rocketchat-seeded` containers and
volume, then restores the baseline again. Any changes made through the UI since
the previous seed are intentionally discarded.

After the first seed, normal start and stop commands are sufficient:

```sh
cd rocket-chat
docker compose up -d
docker compose stop
```

`docker compose up -d` keeps changes made through the UI. It does not re-import
the database JSON when the baseline marker is already present. It does reapply
the preset user passwords from `user-credentials.json`. Use `seed.sh` only when
a deliberate database reset is required.

Useful commands:

```sh
# Show container state
docker compose -p rocketchat-seeded -f rocket-chat/docker-compose.yml ps

# Follow logs
docker compose -p rocketchat-seeded -f rocket-chat/docker-compose.yml logs -f

# Stop without deleting the seeded database
docker compose -p rocketchat-seeded -f rocket-chat/docker-compose.yml stop

# Remove the seeded environment and its isolated volume
docker compose -p rocketchat-seeded -f rocket-chat/docker-compose.yml down -v
```

The safe-state import keeps workspace users, rooms, messages, subscriptions,
roles, permissions, settings, migrations, and avatars. It excludes sessions,
login tokens, scheduler/import history, server events, telemetry/statistics,
workspace credentials, federation keys, and Rocket.Chat Cloud registration
settings. Imported user identifiers and message ownership are not changed.
The dedicated administrator password is stored as the Meteor-compatible bcrypt
hash of its SHA-256 digest; plaintext is not written to MongoDB.

## Preset user credentials

All UI usernames and plaintext local-test passwords are listed in
`user-credentials.json`. The `rocket.cat` bot is intentionally excluded. On
every `docker compose up -d`, the credentials job validates the file, hashes
each password using Rocket.Chat's bundled bcrypt implementation, and updates
the corresponding seeded user. The plaintext JSON is intentionally convenient
for disposable environments and must not be used as a production secret store.
