# Quick Start: Creating Pre-filled Docker Images

## The Problem
Your Docker containers start **empty** every time because data is stored in **volumes** that get recreated.

## The Solution
Create **custom Docker images** with data **baked in** (like TheAgentCompany does).

---

## 3-Step Process

### 1️⃣ Export Data from Running Containers

```bash
cd Arena/apps

# Run your services and populate them with data first!
./start_all_servers.sh

# Then export the data
./export_data_from_running_containers.sh
```

This creates:
- `gitlab/exports/*.tar.gz` - GitLab projects
- `owncloud/owncloud_data/` - OwnCloud files
- `rocketchat/rocketchat.dump` - RocketChat database
- etc.

### 2️⃣ Build Custom Images

```bash
cd Arena/apps

# Build all images at once
./build_all_images.sh
```

This creates:
- `enterpriselab/gitlab:latest`
- `enterpriselab/owncloud:latest`
- `enterpriselab/rocketchat:latest`
- etc.

**Note**: GitLab build takes 15-20 minutes (it actually starts GitLab, imports data, then stops).

### 3️⃣ Update docker-compose.yml

Change from standard images to your custom images:

**Before:**
```yaml
services:
  gitlab:
    image: gitlab/gitlab-ce:latest
    volumes:
      - gitlab_data:/var/opt/gitlab
```

**After:**
```yaml
services:
  gitlab:
    image: enterpriselab/gitlab:latest
    # Remove volumes - data is in the image!
```

Or build on-the-fly:
```yaml
services:
  gitlab:
    build:
      context: .
```

---

## Test It

```bash
# Stop and remove everything
docker-compose down -v

# Start with custom images
docker-compose up -d

# Check - your data should be there! 🎉
```

---

## Push to GitHub (Optional)

Share your pre-filled images:

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Tag images
docker tag enterpriselab/gitlab:latest ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest

# Push
docker push ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest

# Update docker-compose.yml
image: ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
```

---

## Key Files Created

```
Arena/apps/
├── export_data_from_running_containers.sh  ← Step 1: Export data
├── build_all_images.sh                     ← Step 2: Build images
├── QUICK_START.md                          ← This file
├── BUILD_GUIDE.md                          ← Detailed guide
├── DOCKER_IMAGES_README.md                 ← Full documentation
│
├── gitlab/
│   ├── Dockerfile           ← Defines custom GitLab image
│   ├── init.sh              ← Populates data during build
│   ├── exports/             ← Put .tar.gz exports here
│   └── wikis/               ← Put .md wiki pages here
│
└── owncloud/
    ├── Dockerfile           ← Defines custom OwnCloud image
    ├── init.sh              ← Populates data during build
    └── owncloud_data/       ← Put files/folders here
```

---

## How It Works

### Standard Approach (Your Current Setup)
```
┌─────────────────┐
│ Base Image      │  ← gitlab/gitlab-ce:latest
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Docker Volume   │  ← Empty every time! ❌
│ (gitlab_data)   │
└─────────────────┘
```

### Custom Image Approach (Like TheAgentCompany)
```
┌─────────────────────────┐
│ Base Image              │  ← gitlab/gitlab-ce:latest
├─────────────────────────┤
│ + Your data             │  ← Baked into image
│ + Your configuration    │  ← Runs during 'docker build'
├─────────────────────────┤
│ = Custom Image          │  ← enterpriselab/gitlab:latest
└─────────────────────────┘
         ↓
   Always starts with
   your pre-filled data! ✅
```

---

## Troubleshooting

**Build takes forever**
- Normal! GitLab takes 15-20 minutes. It's actually starting the service and importing data.

**Data not persisting**
- Remove volume mounts from docker-compose.yml
- Data should be in the image, not volumes

**Export script can't find data**
- Make sure containers are running: `docker ps`
- Check container names match: `docker ps --format '{{.Names}}'`

---

## What's Different from TheAgentCompany?

| Aspect | TheAgentCompany | Your Setup (Before) | Your Setup (After) |
|--------|----------------|---------------------|-------------------|
| Images | Custom with data | Standard images | Custom with data |
| Data storage | Baked into image | Docker volumes | Baked into image |
| Reproducibility | ✅ Same every time | ❌ Empty on fresh start | ✅ Same every time |
| Distribution | GitHub packages | Local only | GitHub packages |

---

## Summary

✅ Created Dockerfiles for all services  
✅ Created init scripts that populate data during build  
✅ Created export script to extract data from running containers  
✅ Created build script to build all images  
✅ Documented everything thoroughly  

**You now have the same setup as TheAgentCompany!** 🚀

---

## Need More Help?

- Detailed guide: `BUILD_GUIDE.md`
- Full documentation: `DOCKER_IMAGES_README.md`
- Dockerfile examples: Look in `gitlab/` and `owncloud/` directories
- TheAgentCompany reference: https://github.com/TheAgentCompany/TheAgentCompany
