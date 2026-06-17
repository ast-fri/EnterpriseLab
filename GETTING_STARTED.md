# 🏢 EnterpriseLab - Getting Started

**Automated enterprise environment with pre-filled data - ready in minutes!**

---

## 🚀 Quick Start (For New Users)

Clone this repository and run the setup script:

```bash
# Clone the repository
git clone <your-repo-url>
cd EnterpriseLab

# Run automated setup
./SETUP.sh
```

That's it! The setup script will:
- ✅ Check Docker installation
- ✅ Detect pre-filled data backups
- ✅ Pull all necessary images
- ✅ Restore data automatically on first start
- ✅ Start all services

---

## 📦 What's Included

### Services with Pre-filled Data

| Service | Port | Description | Data Included |
|---------|------|-------------|---------------|
| **GitLab** | 8080 | Code repository & CI/CD | Projects, repos, wikis |
| **OwnCloud** | 8081 | File storage & sharing | Files, folders, documents |
| **RocketChat** | 3000 | Team communication | Channels, messages, users |
| **Plane** | 3001 | Project management | Projects, issues, boards |
| **Dolibarr** | 8082 | ERP/CRM system | Customers, invoices, data |

---

## 🎯 Usage

### Start All Services

```bash
cd Arena/apps
./start_all_servers.sh
```

On **first run**, pre-filled data is automatically restored from backups!

### Check Status

```bash
./start_all_servers.sh status
```

### Stop All Services

```bash
./start_all_servers.sh stop
```

---

## 🔐 Default Credentials

### GitLab
- URL: http://localhost:8080
- Username: `root`
- Password: Check your setup or default `admin`

### OwnCloud
- URL: http://localhost:8081
- Username: `admin`
- Password: `admin`

### RocketChat
- URL: http://localhost:3000
- Username: `suraj.nagaje` (or other users from backup)
- Password: *Your configured password* (see note below)

### Plane
- URL: http://localhost:3001
- Create account on first visit or restored from backup

### Dolibarr
- URL: http://localhost:8082
- Username: `admin`
- Password: `admin`

---

## 🛠️ How It Works

### Hybrid Architecture

EnterpriseLab uses a **hybrid approach** for maximum convenience:

1. **Volume Backups** (OwnCloud, Plane, Dolibarr)
   - Pre-filled data stored as compressed volume snapshots
   - Automatically restored on first run
   - Fast and simple

2. **Custom Images** (GitLab - optional)
   - For GitLab with complex project structures
   - Build custom image with your projects
   - Most reproducible approach

### Data Flow

```
Clone Repo → Run SETUP.sh → Start Services
                              ↓
                    First Run Detected
                              ↓
                  Restore Volume Backups
                              ↓
                    Services with Data! ✅
```

---

## 📁 Directory Structure

```
EnterpriseLab/
├── SETUP.sh                          ← Run this first!
├── GETTING_STARTED.md                ← This file
│
└── Arena/apps/
    ├── start_all_servers.sh          ← Start/stop services
    ├── backup_volumes.sh             ← (Maintainer) Create backups
    ├── restore_volumes.sh            ← (Auto) Restore backups
    │
    ├── volume_backups/               ← Pre-filled data backups
    │   ├── owncloud-data.tar.gz
    │   ├── rocketchat-mongodb.tar.gz
    │   └── ...
    │
    ├── gitlab/
    │   └── docker-compose.yml
    ├── owncloud/
    │   └── docker-compose.yml
    ├── rocketchat/
    │   └── docker-compose.yml
    └── ...
```

---

## 🔄 For Maintainers

### Creating Volume Backups

If you're maintaining this repo and want to update the pre-filled data:

```bash
cd Arena/apps

# 1. Start services and populate with your data
./start_all_servers.sh

# ... add projects, files, messages, etc ...

# 2. Create backups
./backup_volumes.sh

# 3. Commit to repo
git add volume_backups/
git commit -m "Update pre-filled data"
git push
```

Now users cloning your repo will get the updated data!

### GitLab Custom Image (Optional)

For GitLab projects:

```bash
cd Arena/apps

# 1. Export GitLab projects
#    - Go to http://localhost:8080
#    - Export each project as .tar.gz
#    - Place in gitlab/exports/

# 2. Build custom image
./setup_gitlab_image.sh

# 3. Update docker-compose.yml to use custom image
# image: enterpriselab/gitlab:latest
```

---

## 🐛 Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker ps

# Check logs
cd Arena/apps
./start_all_servers.sh status
```

### Data not appearing

```bash
# Force volume restore
cd Arena/apps
rm .volumes_restored
./start_all_servers.sh
```

### Port conflicts

Edit `docker-compose.yml` files to change ports:

```yaml
ports:
  - "8080:80"  # Change 8080 to different port
```

### Reset everything

```bash
cd Arena/apps
./start_all_servers.sh stop

# Remove all volumes (⚠️ deletes data!)
docker volume prune

# Start fresh
./start_all_servers.sh
```

---

## 🌟 Features

### ✅ Fully Automated
- One command setup
- Automatic data restoration
- No manual configuration needed

### ✅ Pre-filled Data
- Services ready to use immediately
- Sample projects, files, and data included
- Realistic test environment

### ✅ Reproducible
- Same setup on every machine
- Clone and run
- Perfect for teams

### ✅ Easy Maintenance
- Simple backup script for maintainers
- Version-controlled data
- Easy updates

---

## 📚 Additional Documentation

- **Detailed Guide**: `Arena/apps/BUILD_GUIDE.md`
- **Docker Images**: `Arena/apps/DOCKER_IMAGES_README.md`
- **Quick Reference**: `Arena/apps/QUICK_START.md`

---

## 🤝 Contributing

To contribute updated data:

1. Start services and make your changes
2. Run `./backup_volumes.sh`
3. Commit and push the backups
4. Submit a pull request

---

## 📝 Architecture Notes

This setup uses:
- **Docker Compose** for service orchestration
- **Volume backups** for data persistence
- **Automated restore** on first run
- **Custom images** for complex services (optional)

Inspired by [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) architecture.

---

## ⚡ Performance Tips

### Fast Startup
- Pre-pulled images (done by SETUP.sh)
- Compressed volume backups
- Parallel service startup

### Resource Usage
- Minimum: 8GB RAM, 20GB disk
- Recommended: 16GB RAM, 50GB disk
- All services: ~30-40GB disk usage

### Scaling
- Run individual services: `cd <service> && docker compose up -d`
- Skip large services if not needed
- Adjust resource limits in docker-compose.yml

---

## 🎉 That's It!

You now have a fully functional enterprise environment with:
- ✅ Code repository (GitLab)
- ✅ File storage (OwnCloud)
- ✅ Team chat (RocketChat)
- ✅ Project management (Plane)
- ✅ ERP/CRM (Dolibarr)

All with **pre-filled data** ready to use!

**Happy coding! 🚀**

---

## 📞 Support

- Issues: Open a GitHub issue
- Documentation: See `Arena/apps/` for detailed docs
- Updates: Pull latest changes and re-run `./SETUP.sh`

---

**EnterpriseLab** - Enterprise tools, zero configuration! 🏢
