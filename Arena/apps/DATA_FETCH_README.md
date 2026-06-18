# EnterpriseLab Data Fetching Scripts

This directory contains three scripts for fetching data from hosted applications. Each script uses a different method to access the data.

## 📁 Scripts Overview

### 1. `fetch_data.sh` - API Method (Basic)
Fetches data via public REST APIs and basic authentication.

**Pros:**
- Works without direct database access
- Safe and follows API contracts
- Good for production environments

**Cons:**
- Limited to public endpoints
- Requires valid API credentials
- May not access all data

**Usage:**
```bash
./fetch_data.sh
```

**Outputs:** Metadata and public information from each service.

---

### 2. `fetch_database_data.sh` - Direct Database Access (Recommended)
**⭐ THIS IS THE MAIN SCRIPT YOU SHOULD USE ⭐**

Directly queries application databases (MongoDB, PostgreSQL, MariaDB) to fetch actual application data.

**Pros:**
- ✅ Fetches **real application data** (users, tickets, projects, messages)
- ✅ No authentication required
- ✅ Complete data access
- ✅ Fast and reliable

**Cons:**
- Requires database containers to be running
- Read-only access (which is what we want for visualization)

**Usage:**
```bash
./fetch_database_data.sh
```

**Outputs Real Data:**
- **RocketChat**: Users, rooms, messages, statistics
- **OwnCloud**: Users, files, shares, storage stats
- **GitLab**: Notes about accessing via API
- **Plane**: Workspaces, projects, issues
- **Zammad**: Tickets, users, organizations
- **Dolibarr**: Users, products, contacts

---

### 3. `fetch_data_enhanced.sh` - MCP Server Method (Advanced)
Attempts to query MCP (Model Context Protocol) servers that have direct database access.

**Status:** Experimental - MCP protocol requires session management

**Pros:**
- Designed for programmatic access
- Has built-in authentication

**Cons:**
- Complex session-based protocol
- Requires MCP client library
- Not yet fully implemented

---

## 🚀 Quick Start

### Step 1: Ensure services are running
```bash
cd /mnt/home-ldap/vkharsh_ldap/Research/EnterpriseLab/Arena/apps
./start_all_servers.sh
```

### Step 2: Fetch actual database data (Recommended)
```bash
./fetch_database_data.sh
```

### Step 3: Check the fetched data
```bash
ls -lh fetched_data/*.json
cat fetched_data/summary_database.json
```

---

## 📊 Output Structure

All scripts create JSON files in the `fetched_data/` directory:

```
fetched_data/
├── rocketchat.json         # RocketChat data
├── owncloud.json           # OwnCloud data
├── gitlab.json             # GitLab data
├── plane.json              # Plane data
├── zammad.json             # Zammad data
├── dolibarr.json           # Dolibarr data
├── frappe.json             # Frappe data
└── summary_database.json   # Summary of all fetched data
```

### Example Data Structure

**RocketChat** (`rocketchat.json`):
```json
{
  "service": "rocketchat",
  "url": "http://localhost:3000",
  "database": "mongodb",
  "data": {
    "statistics": {
      "totalUsers": 5,
      "totalRooms": 3,
      "totalMessages": 127
    },
    "users": [...],
    "rooms": [...],
    "recent_messages": [...]
  }
}
```

**Zammad** (`zammad.json`):
```json
{
  "service": "zammad",
  "url": "http://localhost:8083",
  "database": "postgresql",
  "data": {
    "tickets": [
      {
        "id": 7,
        "title": "AC Rattling Noise at Bangalore Office",
        "state_id": 1,
        "priority_id": 2,
        "group_id": 2,
        "created_at": "2026-02-23T18:27:47.564"
      },
      ...
    ],
    "users": [...],
    "organizations": [...]
  }
}
```

---

## 🔧 Troubleshooting

### No data fetched
1. Check if services are running:
   ```bash
   docker ps
   ```

2. Check if database containers exist:
   ```bash
   docker ps | grep -E "mongo|postgres|maria"
   ```

3. Check specific container logs:
   ```bash
   docker logs <container-name>
   ```

### Empty arrays in output
Some applications might not have pre-seeded data. This is normal if:
- The application is freshly installed
- Data hasn't been restored from backups
- No users have created content yet

Example: `"users": []` means there are zero users in the database.

### Permission errors
The scripts need to access Docker containers:
```bash
# Ensure you can run docker commands
docker ps

# If you get permission errors, you may need to be in the docker group
sudo usermod -aG docker $USER
newgrp docker
```

---

## 📈 Visualizing the Data

### Python Example
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load RocketChat data
with open('fetched_data/rocketchat.json') as f:
    rc_data = json.load(f)

# Create statistics dataframe
stats = rc_data['data']['statistics']
df = pd.DataFrame([stats])

# Plot
df.plot(kind='bar')
plt.title('RocketChat Statistics')
plt.tight_layout()
plt.savefig('rocketchat_stats.png')
```

### JavaScript Example
```javascript
const fs = require('fs');

// Load Zammad data
const zammad = JSON.parse(
  fs.readFileSync('fetched_data/zammad.json', 'utf8')
);

// Get ticket counts by state
const tickets = zammad.data.tickets;
const stateCounts = tickets.reduce((acc, ticket) => {
  acc[ticket.state_id] = (acc[ticket.state_id] || 0) + 1;
  return acc;
}, {});

console.log('Tickets by state:', stateCounts);
```

---

## 🗄️ Database Access Details

| Service | Database | Container | Default Port |
|---------|----------|-----------|--------------|
| **RocketChat** | MongoDB | `rocketchat-mongodb-1` | 27017 |
| **OwnCloud** | MariaDB | `owncloud-mariadb-1` | 3306 |
| **Plane** | PostgreSQL | `plane-plane-db-1` | 5432 |
| **Zammad** | PostgreSQL | `zammad-*-postgresql-1` | 5432 |
| **Dolibarr** | MariaDB | `dolibarr-mariadb-1` | 3306 |
| **GitLab** | PostgreSQL | Internal | - |

---

## 🔐 Security Notes

### For Development/Testing
These scripts are safe for local development and testing:
- Read-only database queries
- No data modification
- No credential exposure in output

### For Production
⚠️ **DO NOT use direct database access in production!**
- Use proper API authentication
- Implement rate limiting
- Use read-only database replicas
- Add audit logging
- Encrypt data at rest and in transit

---

## 🎯 Use Cases

1. **Dashboard Creation**: Fetch data periodically and visualize with Grafana/Kibana
2. **Data Analysis**: Export to pandas/R for statistical analysis
3. **Backup Verification**: Check that data is being populated correctly
4. **Integration Testing**: Verify data flows between applications
5. **Reporting**: Generate automated reports on application usage

---

## 📅 Automation

### Run every hour
Add to crontab:
```bash
# Edit crontab
crontab -e

# Add this line (runs at 3 minutes past every hour to avoid :00 load)
3 * * * * cd /mnt/home-ldap/vkharsh_ldap/Research/EnterpriseLab/Arena/apps && ./fetch_database_data.sh >> /tmp/fetch_data.log 2>&1
```

### Run on-demand via API
Create a simple wrapper:
```bash
#!/bin/bash
cd /mnt/home-ldap/vkharsh_ldap/Research/EnterpriseLab/Arena/apps
./fetch_database_data.sh
python3 -m http.server 8000 --directory fetched_data
```

---

## 🤝 Contributing

To add support for a new application:

1. Identify the database type and container
2. Write SQL/NoSQL queries to fetch relevant data
3. Add a `fetch_<appname>_data()` function
4. Call it from `main()`
5. Update this README

---

## 📞 Support

For issues or questions:
- Check container logs: `docker logs <container-name>`
- Verify database connectivity: `docker exec <container> <db-client> --version`
- Review CREDENTIALS.md for authentication details

---

**Last Updated:** June 18, 2026  
**Maintained By:** EnterpriseLab Team
