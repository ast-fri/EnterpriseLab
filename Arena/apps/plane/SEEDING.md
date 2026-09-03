# Seeded Plane

This stack creates an isolated Plane v0.28 database, restores compatible records
from `Extracted_data/plane_from_db.json`, and installs the shared 13-user login
set with preset passwords.

From the repository root:

```bash
./plane/seed.sh Extracted_data/plane_from_db.json plane/docker-compose.yaml
```

The UI is at <http://localhost:3001>. Plane uses email addresses on its login
screen. All emails and passwords are stored in `plane/user-credentials.json`.

The restored baseline contains the Inazuma Engineering workspace, 6 projects,
24 issues, 12 cycles, 18 modules, labels, activities and related preferences.
Every shared user is a member of the workspace and all six projects.

Ordinary starts preserve the seeded database:

```bash
docker compose -p plane-seeded --env-file plane/plane.env \
  -f plane/docker-compose.yaml up -d
```

Run `seed.sh` again to delete only the `plane-seeded` volumes and reconstruct
the baseline.
