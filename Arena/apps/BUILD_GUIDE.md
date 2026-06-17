# Building Pre-filled Docker Images for EnterpriseLab

## Overview

This guide explains how to create Docker images with pre-filled data for your EnterpriseLab applications, similar to TheAgentCompany approach.

## How It Works

1. **Dockerfile with Data Embedding**: Custom Dockerfiles that start from base images and embed your pre-filled data during the build process
2. **Initialization Scripts**: Scripts that run during `docker build` to configure the application and populate data
3. **Persistent Images**: The resulting images contain all the data baked in, so every container starts with the same pre-filled state

## Prerequisites

Before building images, you need to:

1. **Run your services locally** and populate them with the data you want
2. **Export/backup the data** from running containers
3. **Place the exported data** in the appropriate directories

---

## Step-by-Step Process

### 1. GitLab - Export and Build

#### A. Export Projects from Running GitLab

First, export projects from your running GitLab instance:

```bash
# Get a list of projects
curl --header "PRIVATE-TOKEN: your-token" "http://localhost:8080/api/v4/projects"

# Export a specific project (replace PROJECT_ID)
curl --request POST --header "PRIVATE-TOKEN: your-token" \
     "http://localhost:8080/api/v4/projects/PROJECT_ID/export"

# Wait for export to complete, then download
curl --header "PRIVATE-TOKEN: your-token" \
     --output "project-export.tar.gz" \
     "http://localhost:8080/api/v4/projects/PROJECT_ID/export/download"
```

Alternatively, use GitLab UI: Settings → General → Advanced → Export project

#### B. Place Exports in Directory

```bash
# Place your exported .tar.gz files here
cp your-project-export.tar.gz Arena/apps/gitlab/exports/
```

#### C. Add Wiki Pages (Optional)

```bash
# Create markdown files for wiki pages
echo "# Welcome\nThis is the main wiki page." > Arena/apps/gitlab/wikis/home.md
echo "# Getting Started\nHow to get started..." > Arena/apps/gitlab/wikis/getting-started.md
```

#### D. Build GitLab Image

```bash
cd Arena/apps/gitlab
docker build -t enterpriselab/gitlab:latest .
```

**Note**: This build takes 10-20 minutes as it starts GitLab, imports projects, and configures everything.

---

### 2. OwnCloud - Export and Build

#### A. Export Data from Running OwnCloud

```bash
# First, find the running container
docker ps | grep owncloud

# Copy data from running container
docker cp owncloud:/mnt/data/admin Arena/apps/owncloud/owncloud_data/
```

#### B. Build OwnCloud Image

```bash
cd Arena/apps/owncloud
docker build -t enterpriselab/owncloud:latest .
```

---

### 3. RocketChat - Export and Build

#### A. Export MongoDB Database

```bash
# Dump the RocketChat MongoDB database
docker exec rocketchat-mongodb mongodump \
    --db=rocketchat \
    --archive=/tmp/rocketchat.dump

# Copy the dump from container
docker cp rocketchat-mongodb:/tmp/rocketchat.dump ./Arena/apps/rocketchat/
```

#### B. Create Dockerfile for RocketChat

Create `Arena/apps/rocketchat/Dockerfile`:

```dockerfile
FROM registry.rocket.chat/rocketchat/rocket.chat:7.9.6

# Copy database dump
COPY rocketchat.dump /docker-entrypoint-initdb.d/
COPY restore.sh /restore.sh

RUN chmod +x /restore.sh

CMD ["/restore.sh"]
```

Create `Arena/apps/rocketchat/restore.sh`:

```bash
#!/bin/bash
set -e

# Wait for MongoDB to be ready
until mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
    echo "Waiting for MongoDB..."
    sleep 2
done

# Restore database dump
mongorestore --archive=/docker-entrypoint-initdb.d/rocketchat.dump

# Start RocketChat normally
exec node main.js
```

#### C. Build RocketChat Image

```bash
cd Arena/apps/rocketchat
docker build -t enterpriselab/rocketchat:latest .
```

---

## 4. Update docker-compose Files

Update your docker-compose.yml files to use the custom images:

### GitLab docker-compose.yml

```yaml
services:
  gitlab:
    image: enterpriselab/gitlab:latest
    # Or build from local Dockerfile:
    # build:
    #   context: .
    container_name: gitlab
    restart: always
    hostname: localhost
    shm_size: '256m'
    ports:
      - "8080:80"
      - "4430:443"
      - "2222:22"
```

### OwnCloud docker-compose.yml

```yaml
services:
  owncloud:
    image: enterpriselab/owncloud:latest
    # Or build from local Dockerfile:
    # build:
    #   context: .
    container_name: owncloud
    restart: always
    ports:
      - "3001:8080"
```

---

## 5. Push Images to GitHub Container Registry (Optional)

To share your pre-filled images:

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag images
docker tag enterpriselab/gitlab:latest ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
docker tag enterpriselab/owncloud:latest ghcr.io/YOUR_USERNAME/enterpriselab-owncloud:latest

# Push images
docker push ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
docker push ghcr.io/YOUR_USERNAME/enterpriselab-owncloud:latest
```

Then update docker-compose.yml:

```yaml
services:
  gitlab:
    image: ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
    pull_policy: always
```

---

## 6. Build All Images Script

Create `Arena/apps/build_all_images.sh`:

```bash
#!/bin/bash
set -e

echo "Building all EnterpriseLab images..."

# Build GitLab
echo "Building GitLab image..."
cd gitlab && docker build -t enterpriselab/gitlab:latest . && cd ..

# Build OwnCloud
echo "Building OwnCloud image..."
cd owncloud && docker build -t enterpriselab/owncloud:latest . && cd ..

# Build RocketChat
echo "Building RocketChat image..."
cd rocketchat && docker build -t enterpriselab/rocketchat:latest . && cd ..

# Add more services as needed...

echo "All images built successfully!"
echo ""
echo "Images created:"
docker images | grep enterpriselab
```

Make it executable:

```bash
chmod +x Arena/apps/build_all_images.sh
```

---

## Testing Your Images

1. **Stop and remove existing containers**:
   ```bash
   docker-compose down -v
   ```

2. **Start with new images**:
   ```bash
   docker-compose up -d
   ```

3. **Verify data is present**:
   - GitLab: Check if projects are visible at http://localhost:8080
   - OwnCloud: Check if files exist at http://localhost:3001
   - RocketChat: Check if channels/messages exist at http://localhost:3000

---

## Key Differences from TheAgentCompany

| Aspect | TheAgentCompany | Your EnterpriseLab |
|--------|----------------|-------------------|
| Base images | Custom Dockerfiles with data | Standard images + volumes |
| Data persistence | Baked into images | Stored in Docker volumes |
| Initialization | During `docker build` | During `docker run` |
| Reproducibility | ✅ Same state every time | ❌ Empty on fresh start |

---

## Troubleshooting

### Build fails with "service not ready"

The initialization script needs the service running. Increase sleep times in init.sh:

```bash
sleep 30  # instead of sleep 10
```

### Data not persisting

Make sure you're running the initialization during `docker build`, not `docker run`. The `RUN` command in Dockerfile executes during build.

### Image size too large

Docker images with embedded data can be large (5-10 GB). Consider:
- Using `.dockerignore` to exclude unnecessary files
- Cleaning up temporary files in Dockerfile
- Using multi-stage builds

---

## Summary

✅ Created Dockerfiles for GitLab and OwnCloud  
✅ Created initialization scripts that populate data during build  
✅ Images can be pushed to GitHub Container Registry  
✅ Users pulling your images get pre-filled data automatically  

This approach ensures that every time someone starts your EnterpriseLab environment, all the data is already there!
