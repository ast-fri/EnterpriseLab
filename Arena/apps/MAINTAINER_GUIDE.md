# Maintainer Guide - EnterpriseLab

This guide is for maintainers who want to update the pre-filled data in the repository.

---

## Quick Workflow

### 1. Update Your Data

Start services and make changes:

```bash
./start_all_servers.sh

# Then make your changes:
# - Add GitLab projects
# - Upload files to OwnCloud
# - Create RocketChat channels
# - Add Plane projects
# - Configure Dolibarr
# etc.
```

### 2. Create Backups

```bash
# Backup all volumes
./backup_volumes.sh

# This creates compressed snapshots in volume_backups/
```

### 3. Commit and Push

```bash
# Add the new backups
git add volume_backups/

# Commit
git commit -m "Update pre-filled data - $(date +%Y-%m-%d)"

# Push to GitHub
git push origin main
```

### 4. Done!

Users cloning your repo will now get the updated data!

---

## What Gets Backed Up

| Service | Volume | Backup File | Contains |
|---------|--------|-------------|----------|
| **OwnCloud** | Files | `owncloud-data.tar.gz` | User files and folders |
| **OwnCloud** | Database | `owncloud-db.tar.gz` | MySQL database |
| **RocketChat** | MongoDB | `rocketchat-mongodb.tar.gz` | All chat data |
| **Plane** | PostgreSQL | `plane-db.tar.gz` | Projects, issues, data |
| **Plane** | Redis | `plane-redis.tar.gz` | Cache and sessions |
| **Plane** | Uploads | `plane-uploads.tar.gz` | Uploaded files |
| **Dolibarr** | Database | `dolibarr-db.tar.gz` | ERP/CRM data |
| **Dolibarr** | Documents | `dolibarr-documents.tar.gz` | Uploaded documents |
| **GitLab** | Config | `gitlab-config.tar.gz` | Configuration |

---

## Current Backup Sizes

```
total 358M
-rw-r--r-- 1 root root 6.8M dolibarr-db.tar.gz
-rw-r--r-- 1 root root 1.3K dolibarr-documents.tar.gz
-rw-r--r-- 1 root root  53K gitlab-config.tar.gz
-rw-r--r-- 1 root root  36M owncloud-data.tar.gz
-rw-r--r-- 1 root root 5.1M owncloud-db.tar.gz
-rw-r--r-- 1 root root 9.0M plane-db.tar.gz
-rw-r--r-- 1 root root  242 plane-redis.tar.gz
-rw-r--r-- 1 root root 6.2K plane-uploads.tar.gz
-rw-r--r-- 1 root root 301M rocketchat-mongodb.tar.gz
```

**Total: ~358 MB**

---

## GitLab Special Case

GitLab can use either:

### Option A: Volume Backups (Current)
- Backs up GitLab config only
- Does NOT backup project data (would be huge)
- Users start with empty GitLab

### Option B: Custom Image (Recommended for Projects)
If you want to include GitLab projects:

```bash
# 1. Export projects from GitLab UI
#    Settings → General → Advanced → Export project
#    Download .tar.gz files

# 2. Place exports in gitlab/exports/

# 3. Build custom image
./setup_gitlab_image.sh

# 4. Push to Docker Hub or GHCR
docker tag enterpriselab/gitlab:latest ghcr.io/yourusername/enterpriselab-gitlab:latest
docker push ghcr.io/yourusername/enterpriselab-gitlab:latest

# 5. Update gitlab/docker-compose.yml
#    image: ghcr.io/yourusername/enterpriselab-gitlab:latest
```

---

## Git LFS for Large Files

If backups exceed 100MB (Git's warning threshold), use Git LFS:

### Setup Git LFS

```bash
# Install Git LFS
git lfs install

# Track backup files
git lfs track "Arena/apps/volume_backups/*.tar.gz"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Enable Git LFS for volume backups"
git push
```

Now large backups are handled properly!

---

## Updating Individual Services

### OwnCloud Only

```bash
# Backup just OwnCloud
docker run --rm \
    -v owncloud_owncloud_files:/data \
    -v $(pwd)/volume_backups:/backup \
    alpine tar czf /backup/owncloud-data.tar.gz -C / data

# Commit
git add volume_backups/owncloud-data.tar.gz
git commit -m "Update OwnCloud files"
git push
```

### RocketChat Only

```bash
# Backup just RocketChat
docker run --rm \
    -v rocket-chat_mongodb_data:/data \
    -v $(pwd)/volume_backups:/backup \
    alpine tar czf /backup/rocketchat-mongodb.tar.gz -C / data

git add volume_backups/rocketchat-mongodb.tar.gz
git commit -m "Update RocketChat data"
git push
```

---

## Testing Backups

Before committing, test that backups work:

```bash
# 1. Stop services
./start_all_servers.sh stop

# 2. Delete volumes (⚠️ DESTRUCTIVE - make sure backups are good!)
docker volume rm owncloud_owncloud_files owncloud_db_data rocket-chat_mongodb_data

# 3. Restore from backups
./restore_volumes.sh

# 4. Start services
./start_all_servers.sh

# 5. Verify data is present
# - Check OwnCloud files
# - Check RocketChat messages
# - Check Plane projects
# etc.
```

---

## Automation with CI/CD

### GitHub Actions (Example)

Create `.github/workflows/backup.yml`:

```yaml
name: Backup Volumes

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  backup:
    runs-on: self-hosted  # Must run where Docker is available
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: |
          cd Arena/apps
          ./start_all_servers.sh
          sleep 60  # Wait for services to be ready
      
      - name: Create backups
        run: |
          cd Arena/apps
          echo "" | ./backup_volumes.sh
      
      - name: Commit and push
        run: |
          git config user.name "Backup Bot"
          git config user.email "bot@example.com"
          git add Arena/apps/volume_backups/
          git commit -m "Automated backup - $(date +%Y-%m-%d)" || exit 0
          git push
```

---

## Size Optimization

### Compress More

For even smaller backups:

```bash
# Use maximum compression
docker run --rm \
    -v volume_name:/data \
    -v $(pwd)/volume_backups:/backup \
    alpine tar czf9 /backup/file.tar.gz -C / data
```

### Exclude Unnecessary Files

Edit backup commands to exclude:
- Log files
- Cache files
- Temporary files

Example:

```bash
tar czf /backup/file.tar.gz \
    --exclude='*/logs/*' \
    --exclude='*/cache/*' \
    --exclude='*.log' \
    -C / data
```

---

## Versioning Backups

### Tag Releases

```bash
# Create a version tag
git tag -a v1.0.0 -m "EnterpriseLab v1.0.0 with data snapshot"
git push origin v1.0.0
```

Users can then clone specific versions:

```bash
git clone --branch v1.0.0 <repo-url>
```

### Multiple Backup Sets

Create different backup sets:

```
volume_backups/
├── current/           ← Latest data
├── v1.0/             ← Version 1.0 snapshot
└── minimal/          ← Minimal demo data
```

---

## Distribution

### Via Git Repository (Current)
- ✅ Free
- ✅ Version controlled
- ✅ Easy for users
- ⚠️ Size limits (use Git LFS)

### Via Docker Hub/GHCR
- ✅ No size limits
- ✅ Fast downloads
- ⚠️ More complex setup
- ⚠️ Requires building images

### Via Release Assets
- ✅ No repository bloat
- ✅ Multiple versions
- ⚠️ Requires download script

---

## Troubleshooting

### Backup fails

```bash
# Check if services are running
docker ps

# Check if volumes exist
docker volume ls | grep -E "(owncloud|rocket|plane|dolibarr)"

# Check disk space
df -h
```

### Restore fails

```bash
# Check backup files exist
ls -lh volume_backups/

# Check file integrity
tar -tzf volume_backups/owncloud-data.tar.gz > /dev/null

# Try manual restore
docker run --rm \
    -v owncloud_owncloud_files:/data \
    -v $(pwd)/volume_backups:/backup \
    alpine tar xzf /backup/owncloud-data.tar.gz -C /
```

---

## Best Practices

1. **Test backups regularly** - Restore on a clean system to verify
2. **Document changes** - Clear commit messages
3. **Tag versions** - Use semantic versioning
4. **Monitor size** - Keep backups under 500MB if possible
5. **Clean old data** - Remove unnecessary files before backup
6. **Use Git LFS** - For files over 100MB
7. **Automate** - Use CI/CD for regular backups

---

## Quick Reference

```bash
# Full backup workflow
./start_all_servers.sh          # Ensure services running
# ... make changes ...
./backup_volumes.sh             # Create backups
git add volume_backups/         # Stage backups
git commit -m "Update data"     # Commit
git push                        # Push to remote

# Test restore
./start_all_servers.sh stop     # Stop services
docker volume prune -f          # Delete volumes
./restore_volumes.sh            # Restore from backups
./start_all_servers.sh          # Start with restored data

# Individual service backup
docker run --rm -v VOLUME:/data -v $(pwd)/volume_backups:/backup \
    alpine tar czf /backup/FILE.tar.gz -C / data
```

---

## Summary

✅ **Backup**: `./backup_volumes.sh`  
✅ **Commit**: `git add volume_backups/ && git commit && git push`  
✅ **Users**: Clone and run `./SETUP.sh` - data auto-restored!

**That's it! Simple maintenance for reproducible environments.** 🎉
