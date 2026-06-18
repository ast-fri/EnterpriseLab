# 📤 Data Push Script - Restore Data to Applications

This script pushes data from JSON files back into application databases. Useful for:
- **Data Migration**: Moving data between environments
- **Environment Seeding**: Populating test/dev databases
- **Data Restoration**: Restoring from backups
- **Cloning**: Duplicating application setups

---

## 🚀 Quick Start

### 1. Dry Run (Recommended First Step)
Always run in dry-run mode first to see what would be inserted:

```bash
./push_data.sh --dry-run
```

### 2. Actual Data Push
After verifying the dry-run output:

```bash
./push_data.sh
```

You'll be prompted to confirm before any data is inserted.

### 3. Automated Push (No Confirmation)
For scripts/automation:

```bash
AUTO_CONFIRM=1 ./push_data.sh
```

---

## 📋 Command Line Options

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message |
| `--dry-run` | Show what would be done without making changes |
| `--auto-confirm` | Skip confirmation prompt (for automation) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DRY_RUN=1` | Same as `--dry-run` flag |
| `AUTO_CONFIRM=1` | Same as `--auto-confirm` flag |

---

## 📊 What Gets Pushed?

### ✅ RocketChat (MongoDB)
- **Users**: Usernames, emails, roles, status
- **Method**: `updateOne` with `upsert` (no duplicates)
- **Limitation**: Passwords not included (require hashing)

### ✅ Plane (PostgreSQL)
- **Workspaces**: Names, slugs, creation dates
- **Projects**: Names, descriptions, workspace links
- **Issues**: Titles, states, project links
- **Method**: `INSERT ... ON CONFLICT DO NOTHING`

### ⚠️ OwnCloud (MariaDB)
- **Status**: Skipped
- **Reason**: Users require password hashing via `occ` command
- **Alternative**: Use `docker exec owncloud occ user:add <username>`

### ⚠️ Zammad (PostgreSQL)
- **Status**: Skipped
- **Reason**: Tickets require workflow state management
- **Alternative**: Use Zammad REST API for proper ticket creation

### ⚠️ Dolibarr (MariaDB)
- **Products**: Names, prices, references (with "_copy" suffix)
- **Status**: Limited insertion (products only)
- **Reason**: Users require authentication setup
- **Alternative**: Use Dolibarr API for full data migration

### ⚠️ Frappe HRMS (MariaDB)
- **Status**: Skipped
- **Reason**: Requires DocType workflow and naming series
- **Alternative**: Use `bench console` or Frappe API

---

## 🔒 Safety Features

### 1. Dry-Run Mode
The script defaults to showing what would be done:
```bash
DRY_RUN=1 ./push_data.sh
```

### 2. Confirmation Prompt
You must type "yes" to proceed with actual insertion:
```
⚠️  WARNING: This will INSERT data into application databases!
⚠️  Existing data with same IDs will be UPDATED or SKIPPED

Are you sure you want to continue? (yes/no):
```

### 3. No Duplicate IDs
Uses database-specific conflict resolution:
- **MongoDB**: `updateOne` with `upsert`
- **PostgreSQL**: `INSERT ... ON CONFLICT DO NOTHING`
- **MariaDB**: `INSERT ... ON DUPLICATE KEY UPDATE`

### 4. Limited Scope
- Only inserts safe, non-sensitive data
- Skips passwords and authentication data
- Preserves existing records

---

## ⚠️ Important Limitations

### 1. **User Passwords Not Included**
- Passwords require application-specific hashing
- Use application's native user creation tools

### 2. **Complex Workflows Require APIs**
- Tickets, issues, and documents have state machines
- Direct database insertion may bypass validation
- Use REST APIs for production migrations

### 3. **Foreign Key Constraints**
- Related records must exist before insertion
- Script doesn't handle dependency ordering
- May need multiple runs or manual ordering

### 4. **Application-Specific Logic**
- Some apps have triggers, constraints, or business logic
- Direct database insertion bypasses these
- Always test on non-production first

---

## 📝 Example Workflow

### Development Environment Setup

```bash
# 1. Fetch data from production
cd /path/to/production/apps
./fetch_database_data.sh

# 2. Copy JSON files to dev environment
scp -r fetched_data/ dev-server:/path/to/dev/apps/

# 3. Push to dev environment (dry-run first)
cd /path/to/dev/apps
./push_data.sh --dry-run

# 4. Review output, then push for real
./push_data.sh
```

### Data Migration Between Instances

```bash
# Source instance
./fetch_database_data.sh

# Target instance (after transferring JSON files)
# Test first
DRY_RUN=1 ./push_data.sh

# Actual push
AUTO_CONFIRM=1 ./push_data.sh
```

---

## 🔧 Troubleshooting

### Error: "Input directory not found"
```bash
# Run fetch script first
./fetch_database_data.sh
```

### Error: "Container not found"
```bash
# Check if containers are running
docker ps | grep -E "rocketchat|plane|dolibarr"

# Start containers if needed
./start_all_servers.sh
```

### Error: "Permission denied"
```bash
# Check database credentials in script
# Verify container access
docker exec <container> whoami
```

### Foreign Key Violations
```bash
# Insert parent records first
# Example: workspaces before projects
# Or disable FK checks temporarily (not recommended)
```

---

## 🎯 Use Cases

### 1. **Clone Production to Staging**
```bash
# Production
./fetch_database_data.sh
tar -czf data-backup-$(date +%Y%m%d).tar.gz fetched_data/

# Staging
tar -xzf data-backup-*.tar.gz
./push_data.sh --dry-run
./push_data.sh
```

### 2. **Seed Test Environment**
```bash
# Create minimal test dataset
./fetch_database_data.sh

# Edit JSONs to reduce size (optional)
# Keep only test-relevant data

# Push to test DB
DRY_RUN=1 ./push_data.sh
./push_data.sh
```

### 3. **Data Recovery**
```bash
# Restore from backup
cp backup/fetched_data/*.json fetched_data/

# Verify contents
ls -lh fetched_data/

# Restore
./push_data.sh --dry-run
./push_data.sh
```

### 4. **Partial Migration**
```bash
# Fetch from source
./fetch_database_data.sh

# Keep only specific JSONs
rm fetched_data/unwanted_app.json

# Push selected apps
./push_data.sh
```

---

## 🔐 Security Considerations

### ⚠️ For Development/Testing
These defaults are fine for local dev environments.

### ❌ For Production
**DO NOT use this script for production migration without:**

1. ✅ **Full Database Backup**
   ```bash
   ./backup_volumes.sh  # or proper DB dump
   ```

2. ✅ **Use Application APIs**
   - RocketChat: REST API with auth tokens
   - Plane: GraphQL API
   - Zammad: REST API
   - Dolibarr: REST API with modules

3. ✅ **Proper User Management**
   - Don't copy passwords
   - Recreate users via admin panels
   - Setup SSO/LDAP properly

4. ✅ **Validation & Testing**
   - Test on staging first
   - Verify data integrity
   - Check foreign key relationships
   - Validate application functionality

5. ✅ **Audit Trail**
   - Log all operations
   - Document migration steps
   - Keep rollback plan ready

---

## 📈 Advanced Usage

### Custom JSON Source
```bash
# Point to different JSON directory
INPUT_DIR=/path/to/other/jsons ./push_data.sh
```

### Selective Push
Edit the script to comment out unwanted applications:
```bash
# Comment out in main() function:
# push_rocketchat_data || log_warn "RocketChat data push failed"
```

### Modify Before Push
```bash
# Edit JSON files before pushing
vim fetched_data/plane.json

# Then push modified data
./push_data.sh
```

---

## 🤝 Complementary Scripts

| Script | Purpose |
|--------|---------|
| `fetch_database_data.sh` | Extract data FROM databases |
| `push_data.sh` | Insert data INTO databases (this script) |
| `start_all_servers.sh` | Start all application containers |
| `backup_volumes.sh` | Backup Docker volumes |

---

## 📞 Getting Help

```bash
# Show help
./push_data.sh --help

# Check what would be pushed
./push_data.sh --dry-run

# View script source
less push_data.sh
```

---

## ⚠️ Final Warnings

1. **Always backup before pushing data**
2. **Test on non-production first**
3. **Use application APIs for production**
4. **Verify data after insertion**
5. **Keep audit logs of migrations**

---

**Last Updated:** June 18, 2026  
**Maintained By:** EnterpriseLab Team

**Status:** ✅ Functional for development/testing  
**Production Use:** ⚠️ Use with caution, prefer APIs
