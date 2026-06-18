# Fetched Application Data

This directory contains JSON files with data fetched from the EnterpriseLab applications.

## Overview

The `fetch_data.sh` script collects data from all running applications and stores them as individual JSON files for visualization and analysis.

## Generated Files

Each application has its own JSON file:

- **rocketchat.json** - RocketChat team communication data
  - Server info, version, statistics
  - Channels, users, and messages (when authenticated)

- **gitlab.json** - GitLab repository and DevOps data
  - Version information
  - Projects, issues, merge requests

- **owncloud.json** - OwnCloud file sharing data
  - Server status and capabilities
  - User information

- **plane.json** - Plane project management data
  - Health status
  - Projects, issues, and boards (when authenticated)

- **dolibarr.json** - Dolibarr CRM/ERP data
  - API status
  - Business data (when authenticated)

- **zammad.json** - Zammad helpdesk data
  - Container status
  - Ticket information (when accessible)

- **frappe.json** - Frappe ERP data
  - Version information
  - Modules and documents (when authenticated)

- **summary.json** - Summary of all services
  - List of all applications with URLs
  - Fetch timestamp

## Usage

Run the fetch script from the `apps` directory:

```bash
cd /mnt/home-ldap/vkharsh_ldap/Research/EnterpriseLab/Arena/apps
./fetch_data.sh
```

The script will:
1. Check if each service is accessible
2. Fetch available data (public and authenticated endpoints)
3. Save formatted JSON files to `fetched_data/`
4. Create a summary file

## Authentication

Some services require authentication to access full data:

- **RocketChat**: Uses credentials from CREDENTIALS.md (suraj.nagaje / admin123)
- **OwnCloud**: Uses admin/admin credentials
- **GitLab**: Fetches public data only (add personal access token for more data)
- **Plane, Dolibarr, Frappe**: Require API tokens for full access

## Visualizing Data

These JSON files can be used with various visualization tools:

- **Python**: pandas, matplotlib, plotly
- **JavaScript**: D3.js, Chart.js, Apache ECharts
- **BI Tools**: Grafana, Kibana, Tableau

Example Python visualization:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load RocketChat data
with open('fetched_data/rocketchat.json') as f:
    data = json.load(f)

# Extract and visualize
if 'statistics' in data:
    stats = data['statistics']
    print(f"Total Users: {stats.get('totalUsers', 0)}")
    print(f"Total Channels: {stats.get('totalChannels', 0)}")
    print(f"Total Messages: {stats.get('totalMessages', 0)}")
```

## Troubleshooting

If a service shows as "not accessible":

1. Check if the service is running:
   ```bash
   docker ps | grep <service-name>
   ```

2. Verify the port mapping in docker-compose.yml

3. Check service logs:
   ```bash
   docker logs <container-name>
   ```

4. Ensure services are started:
   ```bash
   ./start_all_servers.sh
   ```

## Data Freshness

Data is fetched at the time the script runs. For real-time monitoring:

1. Set up a cron job to run the script periodically
2. Use webhooks for real-time updates
3. Implement streaming data collection

## Last Updated

This directory was created on: 2026-06-18
