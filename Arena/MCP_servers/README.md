# 🔌 MCP Servers

This directory contains **MCP (Model Context Protocol) servers** for various enterprise applications used in the Enterprise Lab / Arena environment.

Each MCP server provides a standardized interface for interacting with a specific application and is deployed using Docker.

---

## 📦 Available MCP Servers

The following MCP servers are included in this directory:

- Aider  
- Dolibarr  
- Frappe  
- GitLab  
- ownCloud  
- Plane  
- Playwright  
- Rocket.Chat  
- Zammad  

---

## ▶️ Starting All MCP Servers

To start **all MCP servers at once**, run the following command from this directory:

```bash
./start_all_servers.sh

```


Ensure the script has execute permissions:

```bash
chmod +x start_all_servers.sh
```

🔐 Authentication Setup (Required)

After starting the application servers, you must:
Create Personal Access Tokens (PATs) for the users created in each application.

Update the corresponding MCP server docker-compose.yml files with these tokens.
The MCP servers rely on these tokens for authenticated communication with the application services.