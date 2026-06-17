# EnterpriseLab Docker Images with Pre-filled Data

## Quick Start

### 1. Populate Your Services with Data

First, run your services and add the data you want to be permanent:

```bash
cd Arena/apps
./start_all_servers.sh
```

Then:
- Add projects to GitLab (localhost:8080)
- Upload files to OwnCloud (localhost:3001)
- Create channels and messages in RocketChat (localhost:3000)
- etc.

### 2. Export Data from Running Containers

#### GitLab

```bash
# Access GitLab at http://localhost:8080
# Login as root with your password
# Go to each project → Settings → General → Advanced → Export project
# Download the .tar.gz files

# Or use the API:
curl --header "PRIVATE-TOKEN: your-token" \
     --output "project1.tar.gz" \
     "http://localhost:8080/api/v4/projects/1/export/download"

# Place exports in the gitlab directory
cp *.tar.gz gitlab/exports/
```

#### OwnCloud

```bash
# Copy data directly from the running container
docker cp owncloud:/mnt/data/admin ./owncloud/owncloud_data/
```

#### RocketChat

```bash
# Find MongoDB container
docker ps | grep mongodb

# Export database
docker exec rocketchat-mongodb mongodump \
    --db=rocketchat \
    --archive=/tmp/rocketchat.dump

# Copy dump
docker cp rocketchat-mongodb:/tmp/rocketchat.dump ./rocketchat/
```

### 3. Build Images with Embedded Data

```bash
cd Arena/apps
./build_all_images.sh
```

This will create:
- `enterpriselab/gitlab:latest`
- `enterpriselab/owncloud:latest`
- `enterpriselab/rocketchat:latest`
- etc.

### 4. Update docker-compose Files

Edit your `docker-compose.yml` files to use the custom images:

**Before:**
```yaml
services:
  gitlab:
    image: gitlab/gitlab-ce:latest
    ...
```

**After:**
```yaml
services:
  gitlab:
    image: enterpriselab/gitlab:latest
    # Or build on-the-fly:
    # build:
    #   context: .
    ...
```

### 5. Test Your Images

```bash
# Stop and remove existing containers
docker-compose down -v

# Start with new images
docker-compose up -d

# Check that your data is there!
```

---

## Detailed Workflow

### Understanding the Problem

When you use standard Docker images with volumes:

```yaml
services:
  gitlab:
    image: gitlab/gitlab-ce:latest
    volumes:
      - gitlab_data:/var/opt/gitlab  # ← Data stored here
```

The `gitlab_data` volume is **empty** every time you create it fresh. Your manually added data is lost.

### The Solution

By creating **custom images** with data embedded during the build process:

1. **Data becomes part of the image layers** (not in volumes)
2. **Every container starts with the same data**
3. **Reproducible across machines**

This is exactly how TheAgentCompany handles their pre-filled servers!

---

## Architecture

### How Custom Images Work

```
┌─────────────────────────────────────────────────────────┐
│ 1. Base Image (e.g., gitlab/gitlab-ce:latest)          │
├─────────────────────────────────────────────────────────┤
│ 2. Copy init script + data                              │
│    COPY init.sh /assets/init.sh                         │
│    COPY exports /assets/exports                         │
├─────────────────────────────────────────────────────────┤
│ 3. Run initialization during BUILD                      │
│    RUN bash /assets/init.sh                             │
│    - Start service temporarily                          │
│    - Import data (projects, files, etc.)                │
│    - Configure settings                                 │
│    - Stop service                                       │
├─────────────────────────────────────────────────────────┤
│ 4. Result: Image with baked-in data                     │
│    Now 'docker run' starts with pre-filled data!        │
└─────────────────────────────────────────────────────────┘
```

### What Happens During `docker build`

When you run `docker build -t enterpriselab/gitlab:latest .`:

1. **Starts from base image** (gitlab/gitlab-ce)
2. **Copies your files** (init.sh, exports/*.tar.gz, wikis/*.md)
3. **Executes RUN commands**:
   - Launches GitLab service
   - Creates admin token
   - Imports projects from .tar.gz files
   - Adds wiki pages
   - Shuts down service
4. **Saves the final state** as a new image

The result: an image where all your data is already present!

---

## File Structure

```
Arena/apps/
├── build_all_images.sh          # Build all images at once
├── BUILD_GUIDE.md               # Detailed guide
├── DOCKER_IMAGES_README.md      # This file
│
├── gitlab/
│   ├── Dockerfile               # Custom GitLab image definition
│   ├── init.sh                  # Initialization script
│   ├── docker-compose.yml       # Updated to use custom image
│   ├── exports/                 # Put .tar.gz project exports here
│   │   └── project1.tar.gz
│   └── wikis/                   # Put .md wiki pages here
│       └── home.md
│
├── owncloud/
│   ├── Dockerfile               # Custom OwnCloud image definition
│   ├── init.sh                  # Initialization script
│   ├── docker-compose.yml       # Updated to use custom image
│   └── owncloud_data/           # Put pre-filled files here
│       └── (files and folders)
│
└── rocketchat/
    ├── Dockerfile               # Custom RocketChat image definition
    ├── restore.sh               # Database restore script
    ├── docker-compose.yml       # Updated to use custom image
    └── rocketchat.dump          # MongoDB dump file
```

---

## Pushing Images to GitHub

To share your pre-filled images publicly:

### 1. Create GitHub Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `write:packages` permission
3. Save the token

### 2. Login to GitHub Container Registry

```bash
echo YOUR_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

### 3. Tag Your Images

```bash
docker tag enterpriselab/gitlab:latest ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
docker tag enterpriselab/owncloud:latest ghcr.io/YOUR_USERNAME/enterpriselab-owncloud:latest
```

### 4. Push to GitHub

```bash
docker push ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
docker push ghcr.io/YOUR_USERNAME/enterpriselab-owncloud:latest
```

### 5. Make Package Public

1. Go to github.com/YOUR_USERNAME?tab=packages
2. Select the package
3. Package settings → Change visibility → Public

### 6. Update docker-compose.yml

```yaml
services:
  gitlab:
    image: ghcr.io/YOUR_USERNAME/enterpriselab-gitlab:latest
    pull_policy: always
```

Now anyone can pull your pre-filled images!

---

## Troubleshooting

### Build Takes Very Long

**Problem**: `docker build` hangs or takes 20+ minutes

**Solution**: This is normal! The build process:
- Starts the actual service (GitLab, OwnCloud, etc.)
- Waits for it to be ready
- Imports/configures data
- Shuts down cleanly

GitLab especially can take 15-20 minutes to build.

---

### Service Won't Start During Build

**Problem**: `init.sh` fails with "connection refused"

**Solution**: Increase wait times in init.sh:

```bash
# Change this:
sleep 10

# To this:
sleep 30
```

---

### Data Not Persisting

**Problem**: Built image but data disappears on restart

**Cause**: You're using volumes in docker-compose.yml that override the image data

**Solution**: Remove volume mounts that conflict:

```yaml
# REMOVE these lines:
volumes:
  - gitlab_data:/var/opt/gitlab
```

The data is already in the image, no need for volumes!

---

### Image Size is Huge

**Problem**: Image is 8-10 GB

**Solution**: This is normal for images with data. To reduce size:

1. **Use .dockerignore**:
```
# .dockerignore
*.log
*.tmp
cache/
```

2. **Clean up in Dockerfile**:
```dockerfile
RUN bash /assets/init.sh && \
    rm -rf /tmp/* && \
    rm -rf /var/log/*
```

3. **Consider multi-stage builds** (advanced)

---

## Advanced: Automated Builds

### GitHub Actions Workflow

Create `.github/workflows/build-images.yml`:

```yaml
name: Build Docker Images

on:
  push:
    branches: [main]
    paths:
      - 'Arena/apps/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push GitLab Image
        run: |
          cd Arena/apps/gitlab
          docker build -t ghcr.io/${{ github.repository_owner }}/enterpriselab-gitlab:latest .
          docker push ghcr.io/${{ github.repository_owner }}/enterpriselab-gitlab:latest

      - name: Build and Push OwnCloud Image
        run: |
          cd Arena/apps/owncloud
          docker build -t ghcr.io/${{ github.repository_owner }}/enterpriselab-owncloud:latest .
          docker push ghcr.io/${{ github.repository_owner }}/enterpriselab-owncloud:latest
```

Now images auto-build on every push!

---

## Comparison: Before vs After

### Before (Current EnterpriseLab)

```yaml
# docker-compose.yml
services:
  gitlab:
    image: gitlab/gitlab-ce:latest  # ← Standard image
    volumes:
      - gitlab_data:/var/opt/gitlab  # ← Empty volume
```

**Result**: Empty GitLab every time you start

---

### After (With Custom Images)

```yaml
# docker-compose.yml
services:
  gitlab:
    image: enterpriselab/gitlab:latest  # ← Custom image
    # No volumes needed!
```

**Result**: Pre-filled GitLab with all your projects!

---

## Key Takeaways

✅ **Dockerfiles define custom images** with your data  
✅ **init.sh scripts run during `docker build`** to populate data  
✅ **Data is baked into image layers**, not stored in volumes  
✅ **Images can be pushed to GitHub** for sharing  
✅ **Every container starts with same pre-filled state**  

This is exactly how TheAgentCompany achieves reproducible, pre-filled environments!

---

## Next Steps

1. ✅ Run your services and populate data
2. ✅ Export data from running containers
3. ✅ Run `./build_all_images.sh`
4. ✅ Update docker-compose.yml files
5. ✅ Test with `docker-compose up -d`
6. ✅ Push to GitHub Container Registry (optional)

For detailed instructions, see `BUILD_GUIDE.md`.

Happy building! 🚀
