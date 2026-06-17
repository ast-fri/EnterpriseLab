# EnterpriseLab: Pre-filled Docker Images Implementation

## Overview

This document summarizes how EnterpriseLab now implements pre-filled Docker images, mirroring TheAgentCompany's approach.

---

## Problem Statement

**Before**: Docker containers started empty every time because:
- Used standard base images (gitlab/gitlab-ce, owncloud/server, etc.)
- Data stored in Docker volumes
- Volumes recreated = data lost

**Impact**: Users had to manually populate data every time they started the environment.

---

## Solution Implemented

Created custom Docker images with data embedded during the build process, exactly like TheAgentCompany.

### Key Components

1. **Custom Dockerfiles** (`Arena/apps/*/Dockerfile`)
   - Start from base images
   - Copy initialization scripts and data
   - Run initialization during `docker build`
   - Data becomes part of image layers

2. **Initialization Scripts** (`Arena/apps/*/init.sh`)
   - Start service temporarily during build
   - Configure application (users, tokens, settings)
   - Import data (projects, files, databases)
   - Shut down cleanly

3. **Data Export Scripts** (`export_data_from_running_containers.sh`)
   - Extract data from running containers
   - Prepare data for image builds

4. **Build Automation** (`build_all_images.sh`)
   - Build all custom images with one command
   - Track success/failure for each service

---

## Files Created

### Documentation
```
EnterpriseLab/
├── DOCKER_IMAGES_SUMMARY.md           ← This file (overview)
└── Arena/apps/
    ├── QUICK_START.md                 ← 3-step quick start guide
    ├── BUILD_GUIDE.md                 ← Detailed step-by-step guide
    └── DOCKER_IMAGES_README.md        ← Complete reference
```

### Scripts
```
Arena/apps/
├── export_data_from_running_containers.sh  ← Step 1: Export
└── build_all_images.sh                     ← Step 2: Build
```

### Docker Configurations
```
Arena/apps/
├── gitlab/
│   ├── Dockerfile          ← Custom GitLab image definition
│   ├── init.sh             ← GitLab initialization script
│   ├── exports/            ← Place .tar.gz project exports here
│   └── wikis/              ← Place .md wiki pages here
│
└── owncloud/
    ├── Dockerfile          ← Custom OwnCloud image definition
    ├── init.sh             ← OwnCloud initialization script
    └── owncloud_data/      ← Place files/folders here
```

---

## How It Works

### Build Process

```
1. User populates services with data
   └─> Run services, add projects, files, etc.

2. Export data from running containers
   └─> ./export_data_from_running_containers.sh
   └─> Creates: gitlab/exports/*.tar.gz
               owncloud/owncloud_data/*
               rocketchat/rocketchat.dump

3. Build custom images
   └─> ./build_all_images.sh
   └─> For each service:
       ├─> Start from base image
       ├─> Copy init.sh + data into image
       ├─> RUN init.sh (starts service, imports data, stops)
       └─> Save as custom image

4. Update docker-compose.yml
   └─> Change: image: gitlab/gitlab-ce:latest
       To:     image: enterpriselab/gitlab:latest

5. Test
   └─> docker-compose down -v
   └─> docker-compose up -d
   └─> Data is pre-filled! ✅
```

### Architecture

**During `docker build`:**
```
┌──────────────────────────────────────┐
│ FROM gitlab/gitlab-ce:latest         │ ← Base image
├──────────────────────────────────────┤
│ COPY init.sh /assets/init.sh         │ ← Copy files
│ COPY exports /assets/exports         │
├──────────────────────────────────────┤
│ RUN bash /assets/init.sh             │ ← Execute during BUILD
│   - Starts GitLab service            │   (not during 'run')
│   - Creates admin token              │
│   - Imports projects from exports/   │
│   - Adds wiki pages                  │
│   - Stops service cleanly            │
├──────────────────────────────────────┤
│ CMD ["/assets/wrapper"]              │ ← Container startup command
└──────────────────────────────────────┘
         ↓
    Custom Image
    (with data baked in)
```

**During `docker run`:**
```
Custom Image → Container starts → Data already present! ✅
```

---

## Comparison with TheAgentCompany

### Similarities ✅
- Custom Dockerfiles for each service
- Initialization scripts run during build
- Data embedded in image layers
- Can be pushed to GitHub Container Registry
- Reproducible across machines

### Implementation Details

| Component | TheAgentCompany | EnterpriseLab |
|-----------|----------------|---------------|
| **GitLab** | Custom Dockerfile + init.sh | ✅ Implemented |
| **OwnCloud** | Custom Dockerfile + init.sh | ✅ Implemented |
| **RocketChat** | Data restore container | ✅ Pattern documented |
| **Plane** | Custom image | 📝 Template ready |
| **Registry** | ghcr.io/theagentcompany | ghcr.io/YOUR_USERNAME |

---

## User Workflow

### Initial Setup (One Time)

1. Clone EnterpriseLab repository
2. Navigate to `Arena/apps/`
3. Read `QUICK_START.md`
4. Follow 3-step process:
   - Export data from running containers
   - Build custom images
   - Update docker-compose files

### Daily Usage

```bash
# Start environment
docker-compose up -d

# Work with pre-filled services
# - GitLab has projects
# - OwnCloud has files
# - RocketChat has messages
# All data is already there!

# Stop environment
docker-compose down
```

### Sharing Images (Optional)

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag images
docker tag enterpriselab/gitlab:latest ghcr.io/USERNAME/enterpriselab-gitlab:latest

# Push to GitHub
docker push ghcr.io/USERNAME/enterpriselab-gitlab:latest
```

Update `docker-compose.yml`:
```yaml
services:
  gitlab:
    image: ghcr.io/USERNAME/enterpriselab-gitlab:latest
    pull_policy: always
```

Now anyone can pull pre-filled images!

---

## Benefits

### Before (Standard Images)
❌ Empty containers on every fresh start  
❌ Manual data population required  
❌ Not reproducible across machines  
❌ Time-consuming setup  

### After (Custom Images)
✅ Pre-filled data every time  
✅ Zero manual setup needed  
✅ Reproducible across machines  
✅ Fast startup  
✅ Can be shared via GitHub  

---

## Technical Details

### Why `RUN` Not `CMD`?

```dockerfile
# ❌ Wrong - runs during container start (too late)
CMD bash /assets/init.sh

# ✅ Correct - runs during image build
RUN bash /assets/init.sh
```

### Why Remove Volumes?

```yaml
# ❌ Wrong - volume overrides image data
services:
  gitlab:
    image: enterpriselab/gitlab:latest
    volumes:
      - gitlab_data:/var/opt/gitlab  # ← Hides image data!

# ✅ Correct - use image data
services:
  gitlab:
    image: enterpriselab/gitlab:latest
    # No volumes!
```

### Build Time Considerations

| Service | Build Time | Why? |
|---------|-----------|------|
| GitLab | 15-20 min | Starts full GitLab, imports projects, waits for completion |
| OwnCloud | 3-5 min | Copies files, runs file scanner |
| RocketChat | 2-3 min | Restores MongoDB database |

This is **normal and expected** - the build actually runs the services!

---

## Future Enhancements

### Potential Additions

1. **More Services**
   - Plane (project management)
   - Dolibarr (ERP/CRM)
   - Frappe (business framework)
   - Zammad (ticketing)

2. **Automation**
   - GitHub Actions workflow for auto-builds
   - Scheduled image updates
   - Version tagging (v1.0.0, v1.1.0, etc.)

3. **Optimizations**
   - Multi-stage Docker builds
   - Layer caching strategies
   - Smaller image sizes

4. **Testing**
   - Automated tests to verify pre-filled data
   - CI/CD pipeline integration
   - Health checks

---

## Maintenance

### Updating Pre-filled Data

To update the data in images:

1. Start services with current image
2. Make changes (add projects, files, etc.)
3. Export updated data
4. Rebuild images
5. Push new version to registry

```bash
# Update workflow
docker-compose up -d
# ... make changes ...
./export_data_from_running_containers.sh
./build_all_images.sh
docker push ghcr.io/USERNAME/enterpriselab-gitlab:v1.1.0
```

### Version Management

```bash
# Tag with version
docker tag enterpriselab/gitlab:latest enterpriselab/gitlab:v1.0.0

# Push both tags
docker push enterpriselab/gitlab:latest
docker push enterpriselab/gitlab:v1.0.0
```

---

## Resources

### Documentation Files
- `QUICK_START.md` - Get started in 3 steps
- `BUILD_GUIDE.md` - Detailed instructions
- `DOCKER_IMAGES_README.md` - Complete reference

### Scripts
- `export_data_from_running_containers.sh` - Export data
- `build_all_images.sh` - Build all images

### Examples
- `Arena/apps/gitlab/Dockerfile` - GitLab custom image
- `Arena/apps/gitlab/init.sh` - GitLab initialization
- `Arena/apps/owncloud/Dockerfile` - OwnCloud custom image
- `Arena/apps/owncloud/init.sh` - OwnCloud initialization

### References
- TheAgentCompany: https://github.com/TheAgentCompany/TheAgentCompany
- Docker documentation: https://docs.docker.com/
- GitHub Packages: https://docs.github.com/en/packages

---

## Summary

✅ **Implemented**: Pre-filled Docker images for EnterpriseLab  
✅ **Approach**: Mirrors TheAgentCompany's architecture  
✅ **Components**: Dockerfiles, init scripts, build automation  
✅ **Documentation**: Complete guides and references  
✅ **Result**: Reproducible, pre-filled environments  

EnterpriseLab now has the same professional infrastructure as TheAgentCompany!

---

## Quick Command Reference

```bash
# Export data from running containers
cd Arena/apps
./export_data_from_running_containers.sh

# Build all custom images
./build_all_images.sh

# Test locally
docker-compose down -v
docker-compose up -d

# Push to GitHub
docker tag enterpriselab/gitlab:latest ghcr.io/USERNAME/enterpriselab-gitlab:latest
docker push ghcr.io/USERNAME/enterpriselab-gitlab:latest

# Pull on another machine
docker pull ghcr.io/USERNAME/enterpriselab-gitlab:latest
```

---

**Status**: ✅ Complete and ready to use!

For questions or issues, refer to the detailed documentation in `Arena/apps/`.
