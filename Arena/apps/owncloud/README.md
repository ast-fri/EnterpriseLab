# Seeded ownCloud

This workflow creates a clean, isolated ownCloud instance, creates the same 13
test users used by the other seeded apps, and reconstructs the exported user
file paths as safe dummy files. The database dump is deliberately not restored
directly because its file-cache records refer to original file bytes that are
not present in the JSON.

## Reset and seed

From the `servers` directory:

```bash
./owncloud/seed.sh Extracted_data/owncloud_from_db.json owncloud/docker-compose.yml
```

Open <http://localhost:8083> and sign in with any account listed in
`owncloud/user-credentials.json`. The administrator is `admin` / `Admin123!`.

The export contains recoverable active file trees for `admin`, `aarav.mittal`,
and `surya.reddy`. Their original paths are recreated. PDF, ODT, PNG, and JPEG
placeholders are valid minimal files; source and text files contain provenance
metadata. The export has no rows in `oc_share`, so there are no shares to
reconstruct.

## Start and stop without resetting

```bash
docker compose -p owncloud-seeded -f owncloud/docker-compose.yml up -d
docker compose -p owncloud-seeded -f owncloud/docker-compose.yml down
```

The named volumes remain intact after `down`, so a normal `up -d` reuses the
seeded users and files. Running `seed.sh` intentionally executes `down -v` and
creates a fresh baseline.
