# 🧪 Enterprise Lab – Applications Overview

This directory contains the set of enterprise applications used in the **Enterprise Lab** environment.  
Each application is provided with its own configuration and setup instructions.

---

## 📦 Available Applications

The following applications are available in this lab:

- **Dolibarr**  
  CRM system  
  📁 `dolibarr/`

- **Frappe**  
  ERM system  
  📁 `frappe/`

- **GitLab**  
  Source code management and DevOps platform  
  📁 `gitlab/`

- **ownCloud**  
  File sharing and collaboration platform  
  📁 `owncloud/`

- **Plane**  
  Project management and issue tracking tool  
  📁 `plane/`

- **Rocket.Chat**  
  Team communication and collaboration platform  
  📁 `rocketchat/`

- **Zammad**  
  Helpdesk and ticketing system  
  📁 `zammad/`

Each application directory contains a `README.md` with app-specific setup instructions.  
Some applications also include a `docker-compose.yml` file for containerized deployment.

---

## ▶️ Starting All Application Servers

To start **all application servers at once**, run the following command from this directory:

```bash
./start_all_servers.sh
```


Ensure the script has execute permissions:

```bash
chmod +x start_all_servers.sh
```