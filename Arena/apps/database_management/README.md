# Database Management Scripts

Simple scripts to fetch, modify, and sync data across all EnterpriseLab applications without using the UI.

---

## 📁 Files

| File | Purpose |
|------|---------|
| `fetch_database_data.sh` | Extract ALL data from application databases |
| `fetch_database_schema.sh` | Extract complete database schemas (tables + columns) |
| `fetch_schema_dolibarr_frappe.sh` | Helper for Dolibarr & Frappe schemas (large databases) |
| `sync_data.sh` | Push modified data back to databases (INSERT + UPDATE) |

---

## 🚀 Quick Start

### 1️⃣ Fetch All Data
```bash
./fetch_database_data.sh
```
**Output:** `../fetched_data/*.json` (all application data)

### 2️⃣ Edit Data
```bash
vim ../fetched_data/plane.json          # Edit projects, issues
vim ../fetched_data/rocketchat.json     # Edit users, messages
vim ../fetched_data/dolibarr.json       # Edit products, invoices
```

### 3️⃣ Sync Changes Back
```bash
./sync_data.sh --dry-run    # Preview changes
./sync_data.sh              # Apply changes
```

Done! Changes applied to databases without touching the UI.

---

## 📖 Detailed Usage

### `fetch_database_data.sh`

**Purpose:** Extract complete data from all running applications

**What it fetches:**
- RocketChat: Users, rooms, messages, subscriptions, roles, integrations
- Plane: Workspaces, projects, issues, cycles, modules, assignees
- Dolibarr: Users, companies, products, invoices, orders, proposals, projects
- OwnCloud: Users, files, shares, groups, activity, comments
- Zammad: Tickets, articles, users, groups, roles, tags
- Frappe: Employees, departments, attendance, leaves, salaries, shifts, holidays

**Command:**
```bash
./fetch_database_data.sh
```

**Output Location:** `../fetched_data/`
- `rocketchat.json` (~22 MB - all 1,261 users, 2,136 rooms, 39,326 messages)
- `plane.json` (~60 KB - all 34 projects, 65 issues)
- `dolibarr.json` (~301 KB - all 1,258 users, 33 companies)
- `owncloud.json` (~27 KB)
- `zammad.json` (~5 KB)
- `frappe.json` (~16 KB)
- `gitlab.json` (pending - container upgrading)

**Features:**
- ✅ No limits - fetches ALL data entries
- ✅ Direct database access (bypasses APIs)
- ✅ JSON format (easy to edit)
- ✅ Includes statistics

---

### `fetch_database_schema.sh`

**Purpose:** Extract database structure (all tables and their columns)

**What it extracts:**
- Table names
- Column names, data types, constraints
- Primary keys, foreign keys
- Default values, nullable fields
- All column metadata

**Command:**
```bash
./fetch_database_schema.sh
```

**Output Location:** `../fetched_schemas/`
- `rocketchat_schema.json` (85 collections)
- `plane_schema.json` (107 tables, 1,499 columns)
- `dolibarr_schema.json` (50 tables, ~290 KB)
- `owncloud_schema.json` (51 tables, 321 columns)
- `zammad_schema.json` (118 tables, 1,084 columns)
- `frappe_schema.json` (100 DocTypes)

**Format:**
```json
{
  "tables": [
    {
      "name": "workspaces",
      "columns": [
        {
          "column_name": "id",
          "data_type": "uuid",
          "is_nullable": "NO",
          "column_key": "PRI",
          "ordinal_position": 1,
          ...
        }
      ]
    }
  ]
}
```

---

### `fetch_schema_dolibarr_frappe.sh`

**Purpose:** Extract schemas for Dolibarr & Frappe (optimized for large databases)

**Why separate script?**
- Dolibarr: 403 tables (too large for single query)
- Frappe: 855 DocTypes (would create huge file)
- Solution: Fetches first 50-100 most important tables

**Command:**
```bash
./fetch_schema_dolibarr_frappe.sh
```

**Output:**
- Updates `../fetched_schemas/dolibarr_schema.json` (50 main tables)
- Updates `../fetched_schemas/frappe_schema.json` (100 main DocTypes)

---

### `sync_data.sh`

**Purpose:** Push modified JSON data back to databases

**Operations:**
- ✅ **INSERT** - Add new records
- ✅ **UPDATE** - Modify existing records
- ❌ **DELETE** - Manual SQL required (safety)

**Command:**
```bash
# Dry run (preview changes)
./sync_data.sh --dry-run

# Apply changes
./sync_data.sh

# Skip confirmation (automation)
AUTO_CONFIRM=1 ./sync_data.sh
```

**How it works:**
- MongoDB: `replaceOne({_id}, data, {upsert: true})`
- PostgreSQL: `INSERT ... ON CONFLICT DO UPDATE`
- MariaDB: `INSERT ... ON DUPLICATE KEY UPDATE`

**Supported Apps:**
- ✅ RocketChat (users, rooms, messages)
- ✅ Plane (workspaces, projects, issues)
- ✅ Dolibarr (users, products)
- ⚠️ OwnCloud (requires `occ` command)
- ⚠️ Zammad (requires API for tickets)
- ⚠️ Frappe (requires `bench` or API)

---

## 📝 Complete Workflow Example

### Example 1: Update Project Names in Plane
```bash
# 1. Fetch data
./fetch_database_data.sh

# 2. Edit project names
vim ../fetched_data/plane.json
# Change: "name": "Old Project" → "name": "New Project"

# 3. Preview changes
./sync_data.sh --dry-run

# 4. Apply changes
./sync_data.sh

# Done! Project renamed in Plane without UI
```

### Example 2: Bulk Add RocketChat Users
```bash
# 1. Fetch existing users
./fetch_database_data.sh

# 2. Add new users to JSON
vim ../fetched_data/rocketchat.json
# Add new user objects to "users" array

# 3. Sync to database
./sync_data.sh

# Done! New users added
```

### Example 3: Update Product Prices in Dolibarr
```bash
# 1. Fetch products
./fetch_database_data.sh

# 2. Update prices
vim ../fetched_data/dolibarr.json
# Modify "price" fields

# 3. Sync changes
./sync_data.sh

# Done! Prices updated in Dolibarr
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `0` | Set to `1` for dry-run mode |
| `AUTO_CONFIRM` | `0` | Set to `1` to skip confirmation |

**Examples:**
```bash
# Dry run mode
DRY_RUN=1 ./sync_data.sh

# Auto-confirm (scripts/automation)
AUTO_CONFIRM=1 ./sync_data.sh
```

---

## 🔒 Safety Features

### Data Fetching
- ✅ Read-only operations
- ✅ No database modifications
- ✅ Safe to run anytime

### Data Syncing
- ✅ Dry-run mode available
- ✅ Confirmation prompt (unless auto-confirm)
- ✅ Upsert logic (no duplicates)
- ✅ Preserves existing data
- ⚠️ No automatic DELETE (manual only)

---

## 📊 Data Volumes

| Application | Records | File Size |
|-------------|---------|-----------|
| RocketChat  | 1,261 users, 2,136 rooms, 39,326 messages | 22 MB |
| Plane       | 34 projects, 65 issues | 60 KB |
| Dolibarr    | 1,258 users, 33 companies, 20 invoices | 301 KB |
| OwnCloud    | 4 users, 159 files | 27 KB |
| Zammad      | 8 tickets, 6 users | 5 KB |
| Frappe      | 2 employees, 27 departments | 16 KB |

**Total:** ~22.4 MB of complete application data

---

## ❗ Important Notes

### What's Included
- ✅ ALL data entries (no limits)
- ✅ ALL columns (no filtering)
- ✅ Complete schemas (all tables)
- ✅ Relationships preserved

### What's NOT Included
- ❌ Passwords (require proper hashing)
- ❌ Binary files (images, attachments)
- ❌ Encrypted fields
- ❌ System-internal tables

### Limitations
- **Delete operations:** Must be done manually via SQL
- **File uploads:** Not synced (only metadata)
- **Complex workflows:** Some apps need API (Zammad tickets, Frappe DocTypes)

---

## 🛠️ Troubleshooting

### "Container not found"
```bash
# Check if containers are running
docker ps | grep -E "rocketchat|plane|dolibarr"

# Start containers if needed
cd .. && ./start_all_servers.sh
```

### "Permission denied"
```bash
# Make scripts executable
chmod +x *.sh
```

### "JSON parse error"
```bash
# Validate JSON
python3 -m json.tool ../fetched_data/plane.json

# Fix syntax errors in your editor
```

### "Database not accessible"
```bash
# Check database credentials in scripts
# Verify container names:
docker ps --format "table {{.Names}}\t{{.Image}}"
```

---

## 📂 Directory Structure

```
apps/
├── database_management/          (this folder)
│   ├── README.md                 (this file)
│   ├── fetch_database_data.sh    (fetch all data)
│   ├── fetch_database_schema.sh  (fetch schemas)
│   ├── fetch_schema_dolibarr_frappe.sh  (helper)
│   └── sync_data.sh              (push changes)
├── fetched_data/                 (data output)
│   ├── rocketchat.json
│   ├── plane.json
│   ├── dolibarr.json
│   ├── owncloud.json
│   ├── zammad.json
│   └── frappe.json
└── fetched_schemas/              (schema output)
    ├── rocketchat_schema.json
    ├── plane_schema.json
    └── ...
```

---

## 🎯 Use Cases

### Development
- Clone production data to dev environment
- Test with real data
- Debug issues with actual records

### Testing
- Seed test databases
- Create test scenarios
- Validate data migrations

### Backup & Restore
- Regular data backups
- Point-in-time recovery
- Cross-environment sync

### Bulk Operations
- Mass updates without UI
- Data cleanup scripts
- Bulk imports/exports

---

## ✅ Quick Reference

```bash
# Fetch everything
./fetch_database_data.sh
./fetch_database_schema.sh

# Edit data
vim ../fetched_data/plane.json

# Preview changes
./sync_data.sh --dry-run

# Apply changes
./sync_data.sh

# Automation
AUTO_CONFIRM=1 ./sync_data.sh
```

---

**Last Updated:** June 18, 2026  
**Status:** ✅ Operational - All limits removed, full data extraction
