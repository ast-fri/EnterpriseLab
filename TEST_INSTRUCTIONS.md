# 🧪 Testing Instructions

## Pre-Test Checklist

Before testing, ensure:
- ✅ No hardcoded local paths (checked!)
- ✅ All scripts use relative paths (checked!)
- ✅ Volume backups exist (358 MB)
- ✅ All services are currently running

---

## Test Scenario: Simulate New User Experience

### Step 1: Stop Current Services

```bash
cd /mnt/home-ldap/ankush_ldap/EnterpriseLab/Arena/apps
./start_all_servers.sh stop
```

### Step 2: Create Test Clone

Simulate a fresh clone in a different directory:

```bash
# Go to your home directory
cd /mnt/home-ldap/ankush_ldap

# Create a test clone
cp -r EnterpriseLab EnterpriseLab_TEST

# Enter test directory
cd EnterpriseLab_TEST
```

### Step 3: Clean Test Environment

Remove any state that might exist:

```bash
# Remove the "already restored" marker if it exists
rm -f Arena/apps/.volumes_restored

# Verify backups are present
ls -lh Arena/apps/volume_backups/
```

**Expected output:**
```
total 358M
-rw-r--r-- 1 ... dolibarr-db.tar.gz
-rw-r--r-- 1 ... dolibarr-documents.tar.gz
-rw-r--r-- 1 ... gitlab-config.tar.gz
-rw-r--r-- 1 ... owncloud-data.tar.gz
-rw-r--r-- 1 ... owncloud-db.tar.gz
-rw-r--r-- 1 ... plane-db.tar.gz
-rw-r--r-- 1 ... plane-redis.tar.gz
-rw-r--r-- 1 ... plane-uploads.tar.gz
-rw-r--r-- 1 ... rocketchat-mongodb.tar.gz
```

### Step 4: Run Setup Script

```bash
# Make sure it's executable
chmod +x SETUP.sh

# Run setup
./SETUP.sh
```

**What to expect:**
1. ✅ Docker check passes
2. ✅ Detects pre-filled data (358 MB)
3. ✅ Shows list of available backups
4. ✅ Pulls any missing images
5. ✅ Asks if you want to start services

**Choose:** Type `n` (No) for now - we'll start manually to see the restore

### Step 5: Start Services and Watch Restore

```bash
cd Arena/apps

# Start services (this will trigger volume restore)
./start_all_servers.sh
```

**What to expect:**
```
🔄 First run detected - restoring pre-filled data...

======================================
Restore Docker Volumes
======================================

Restoring OwnCloud Files...
✓ OwnCloud Files restored

Restoring OwnCloud Database...
✓ OwnCloud Database restored

Restoring RocketChat MongoDB...
✓ RocketChat MongoDB restored

Restoring Plane Database...
✓ Plane Database restored

... (etc)

======================================
Volume Restore Complete!
======================================

Your services now have pre-filled data!

🚀 Starting all enterprise servers...
➡️  Starting gitlab ...
➡️  Starting owncloud ...
... (services starting)
✅ All servers started successfully!
```

### Step 6: Verify Services Have Data

Wait 1-2 minutes for services to fully start, then check:

**1. Check Docker containers:**
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

**2. Check OwnCloud (should have files):**
```bash
# Open in browser or check via API
curl -u admin:admin http://localhost:8081/remote.php/dav/files/admin/ | head -50
```

**3. Check RocketChat (should have data):**
```bash
# Open browser
xdg-open http://localhost:3000 &
# Or check container
docker logs rocket-chat-rocketchat-1 | tail -20
```

**4. Check GitLab:**
```bash
# Open browser
xdg-open http://localhost:8080 &
```

**5. Check Plane:**
```bash
xdg-open http://localhost:3001 &
```

**6. Check Dolibarr:**
```bash
xdg-open http://localhost:8082 &
```

### Step 7: Verify Second Start Skips Restore

Stop and start again:

```bash
# Stop services
./start_all_servers.sh stop

# Start again
./start_all_servers.sh
```

**What to expect:**
- ✅ Should NOT restore volumes again
- ✅ Message: "Volumes already restored"
- ✅ Services start immediately
- ✅ Data persists

### Step 8: Test Force Restore

```bash
# Stop services
./start_all_servers.sh stop

# Delete the marker to force restore
rm .volumes_restored

# Start services
./start_all_servers.sh
```

**What to expect:**
- ✅ Restores volumes again
- ✅ Creates new marker file
- ✅ Data is refreshed

---

## Test Scenario: Volume Restore from Scratch

### Clean Slate Test

```bash
cd /mnt/home-ldap/ankush_ldap/EnterpriseLab_TEST/Arena/apps

# Stop everything
./start_all_servers.sh stop

# DELETE VOLUMES (⚠️ Destructive - but we have backups!)
docker volume rm owncloud_owncloud_files owncloud_db_data 2>/dev/null || true
docker volume rm rocket-chat_mongodb_data 2>/dev/null || true
docker volume rm plane_pgdata plane_redisdata plane_uploads 2>/dev/null || true
docker volume rm dolibarr_dolibarr_db_data dolibarr_dolibarr_documents 2>/dev/null || true
docker volume rm gitlab_gitlab_config 2>/dev/null || true

# Remove restore marker
rm -f .volumes_restored

# Start services (should restore from backups)
./start_all_servers.sh
```

**What to expect:**
- ✅ Volumes are recreated from backups
- ✅ All data is restored
- ✅ Services start with pre-filled data

---

## Test Scenario: Manual Backup and Restore

### Test Backup Script

```bash
cd /mnt/home-ldap/ankush_ldap/EnterpriseLab_TEST/Arena/apps

# Ensure services are running
./start_all_servers.sh status

# Create a test backup
echo "" | ./backup_volumes.sh

# Check new backups
ls -lh volume_backups/
```

**What to expect:**
- ✅ All services backed up successfully
- ✅ Backup files in volume_backups/
- ✅ Total ~358 MB

### Test Restore Script Directly

```bash
# Run restore script directly
./restore_volumes.sh
```

**What to expect:**
- ✅ Skips volumes that already have data
- ✅ Only restores empty volumes
- ✅ No errors

---

## Verification Checklist

After testing, verify:

- [ ] ✅ OwnCloud has files visible
- [ ] ✅ RocketChat has channels/messages
- [ ] ✅ Plane has projects/issues
- [ ] ✅ Dolibarr has data
- [ ] ✅ GitLab configuration present
- [ ] ✅ Second start skips restore
- [ ] ✅ Force restore works
- [ ] ✅ Volume deletion + restore works
- [ ] ✅ Backup script creates new backups
- [ ] ✅ No hardcoded paths in errors

---

## Common Issues and Solutions

### Issue: "Volume already has data, skipping..."

**Cause:** Volume has existing data  
**Solution:** This is normal! It means data is already there.

### Issue: Backup script fails

**Cause:** Services not running  
**Solution:** `./start_all_servers.sh` first

### Issue: Restore takes forever

**Cause:** Large RocketChat backup (301 MB)  
**Solution:** Wait 30-60 seconds, it's normal

### Issue: Services don't show data

**Cause:** Need time to fully start  
**Solution:** Wait 2-3 minutes, especially for GitLab

---

## Clean Up Test Environment

After successful testing:

```bash
cd /mnt/home-ldap/ankush_ldap

# Stop test services
cd EnterpriseLab_TEST/Arena/apps
./start_all_servers.sh stop

# Remove test directory
cd /mnt/home-ldap/ankush_ldap
rm -rf EnterpriseLab_TEST

# Restart original services
cd EnterpriseLab/Arena/apps
./start_all_servers.sh
```

---

## What Success Looks Like

✅ **Setup script runs cleanly**  
✅ **Volume restore happens on first start**  
✅ **All services start with pre-filled data**  
✅ **Second start skips restore**  
✅ **Services show actual data when accessed**  
✅ **No hardcoded paths in output**  
✅ **No errors in logs**  

---

## Ready to Commit?

If all tests pass:

```bash
cd /mnt/home-ldap/ankush_ldap/EnterpriseLab

# Check what's new
git status

# Add everything
git add Arena/apps/volume_backups/
git add Arena/apps/*.sh Arena/apps/*.md
git add SETUP.sh *.md .gitignore

# Commit
git commit -m "Add automated pre-filled data system

- Hybrid approach: volume backups + optional custom images
- One-command setup for users (./SETUP.sh)
- Automatic volume restoration on first run
- 358 MB of pre-filled data for all services
- Comprehensive documentation (8 guides)
- Tested and verified working
"

# Push to GitHub
git push origin main
```

---

## Post-Push Test

After pushing to GitHub, test the real clone experience:

```bash
# On another machine or directory
cd /tmp

# Clone your repo
git clone <your-github-repo-url>
cd EnterpriseLab

# Run setup
./SETUP.sh

# Verify data is present
cd Arena/apps
./start_all_servers.sh status
```

**This is what your users will experience!**

---

## Quick Test Commands Summary

```bash
# Quick test from test clone
cd /mnt/home-ldap/ankush_ldap
cp -r EnterpriseLab EnterpriseLab_TEST
cd EnterpriseLab_TEST
rm -f Arena/apps/.volumes_restored
./SETUP.sh

# Verify
cd Arena/apps
./start_all_servers.sh
sleep 60  # Wait for services
docker ps
xdg-open http://localhost:8081  # Check OwnCloud
xdg-open http://localhost:3000  # Check RocketChat

# Clean up
cd /mnt/home-ldap/ankush_ldap
./EnterpriseLab_TEST/Arena/apps/start_all_servers.sh stop
rm -rf EnterpriseLab_TEST
```

---

**Ready to test?** Start with Step 1! 🧪
