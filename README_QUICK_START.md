# 🚀 EnterpriseLab - Quick Start

> **Want to get started in 2 minutes?** This is for you!  
> For the full documentation, see [README.md](README.md)

---

## One-Command Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd EnterpriseLab
./SETUP.sh
```

**That's it!** The setup script will:
- ✅ Check Docker installation
- ✅ Pull all required images
- ✅ Restore pre-filled data automatically
- ✅ Start all services

---

## What You Get

| Service | URL | Credentials | Pre-filled Data |
|---------|-----|-------------|-----------------|
| **GitLab** | http://localhost:8080 | root / admin | Projects & repos |
| **OwnCloud** | http://localhost:8081 | admin / admin | Files & folders |
| **RocketChat** | http://localhost:3000 | From backup | Messages & channels |
| **Plane** | http://localhost:3001 | From backup | Projects & issues |
| **Dolibarr** | http://localhost:8082 | admin / admin | CRM data |

---

## Common Commands

```bash
cd Arena/apps

# Start all services
./start_all_servers.sh

# Check status
./start_all_servers.sh status

# Stop all services
./start_all_servers.sh stop

# Force restore data
rm .volumes_restored && ./start_all_servers.sh
```

---

## Folder Structure

```
EnterpriseLab/
├── SETUP.sh                   ← Run this first!
├── GETTING_STARTED.md         ← Detailed guide
├── README.md                  ← Full documentation
│
└── Arena/apps/
    ├── start_all_servers.sh   ← Daily usage
    ├── volume_backups/        ← Pre-filled data (358MB)
    │
    ├── gitlab/                ← Services
    ├── owncloud/
    ├── rocketchat/
    ├── plane/
    └── dolibarr/
```

---

## For Maintainers

Update the pre-filled data:

```bash
cd Arena/apps

# 1. Make your changes in the running services
./start_all_servers.sh

# 2. Backup volumes
./backup_volumes.sh

# 3. Commit
git add volume_backups/
git commit -m "Update pre-filled data"
git push
```

See [MAINTAINER_GUIDE.md](Arena/apps/MAINTAINER_GUIDE.md) for details.

---

## Requirements

- **Docker** & **Docker Compose**
- **8GB RAM** minimum (16GB recommended)
- **50GB disk** space

---

## Troubleshooting

**Services won't start?**
```bash
docker ps                    # Check Docker is running
./start_all_servers.sh status
```

**Data not appearing?**
```bash
rm .volumes_restored         # Force restore
./start_all_servers.sh
```

**Port conflicts?**
```bash
# Edit docker-compose.yml to change ports
cd gitlab
nano docker-compose.yml      # Change 8080:80 to different port
```

---

## Architecture

**Hybrid Approach** for easy reproducibility:
- 📦 **Volume Backups** for OwnCloud, RocketChat, Plane, Dolibarr
- 🐳 **Custom Images** for GitLab (optional)
- 🔄 **Automatic Restore** on first run

Inspired by [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany).

---

## More Documentation

- 📖 [GETTING_STARTED.md](GETTING_STARTED.md) - Complete user guide
- 🔧 [Arena/apps/MAINTAINER_GUIDE.md](Arena/apps/MAINTAINER_GUIDE.md) - For maintainers
- 📚 [README.md](README.md) - Full documentation

---

**Ready to go!** 🎉

Start with `./SETUP.sh` and you'll have a working enterprise environment in minutes!
