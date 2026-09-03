# Seeded GitLab

This stack initializes GitLab CE 18.5.7 with the database-backed state in
`../Extracted_data/gitlab_from_db.json` and sets deterministic UI passwords from
`user-credentials.json`.

## Fresh reset and seed

From the repository root:

```bash
./gitlab/seed.sh
```

This deliberately removes **only** the Compose project named `gitlab-seeded`,
including its three isolated volumes. It does not reuse or delete volumes from
the old Compose project named `gitlab`.

Initial GitLab startup commonly takes 10-20 minutes. The script waits until the
database import, password setup, and record-count checks all finish.

Open <http://localhost:8081> and use an account from
`gitlab/user-credentials.json`. `aarav.mittal` is the seeded administrator.

## Normal restart

After the first successful seed, this preserves the seeded database:

```bash
docker compose -f gitlab/docker-compose.yaml up -d
```

The seed service recognizes the existing import and only reapplies the known
passwords. To return to the original exported state, run `./gitlab/seed.sh`.

## Export limitation

The JSON is a PostgreSQL table export, not a GitLab backup. It does not contain
Git repository object storage or uploaded file bodies. The imported project
shells are therefore retained only as hidden legacy records. Seeding creates six
clean GitLab-managed projects at the intended paths, initializes each repository
with a README, and restores the intended user memberships. The original commits,
branches, issues, merge requests, and attachment bodies are not copied into
these replacement repositories. Use `gitlab-backup` to preserve those parts in
a future export.

## Diagnostics

```bash
docker compose -f gitlab/docker-compose.yaml ps
docker compose -f gitlab/docker-compose.yaml logs seed
docker compose -f gitlab/docker-compose.yaml logs --tail=200 gitlab
```
