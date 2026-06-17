#!/bin/bash

# Script to export data from running EnterpriseLab containers
# Run this BEFORE building custom images

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Export Data from Running Containers${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if container is running
check_container() {
    local container_name=$1
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

# Export GitLab data
export_gitlab() {
    echo -e "${YELLOW}Exporting GitLab data...${NC}"

    if ! check_container "gitlab"; then
        echo -e "${RED}GitLab container not running. Start it first with: cd gitlab && docker-compose up -d${NC}"
        return 1
    fi

    mkdir -p gitlab/exports
    mkdir -p gitlab/wikis

    echo ""
    echo -e "${GREEN}To export GitLab projects:${NC}"
    echo ""
    echo "Option 1: Using GitLab UI"
    echo "  1. Go to http://localhost:8080"
    echo "  2. Login as root"
    echo "  3. For each project: Settings → General → Advanced → Export project"
    echo "  4. Download the .tar.gz files"
    echo "  5. Move them to: $SCRIPT_DIR/gitlab/exports/"
    echo ""
    echo "Option 2: Using GitLab API"
    echo "  # First, get your access token from GitLab UI (Preferences → Access Tokens)"
    echo "  # Then run these commands:"
    echo ""
    echo "  # List all projects:"
    echo "  curl --header \"PRIVATE-TOKEN: your-token\" http://localhost:8080/api/v4/projects"
    echo ""
    echo "  # Export project (replace PROJECT_ID):"
    echo "  curl --request POST --header \"PRIVATE-TOKEN: your-token\" \\"
    echo "       http://localhost:8080/api/v4/projects/PROJECT_ID/export"
    echo ""
    echo "  # Check export status:"
    echo "  curl --header \"PRIVATE-TOKEN: your-token\" \\"
    echo "       http://localhost:8080/api/v4/projects/PROJECT_ID/export"
    echo ""
    echo "  # Download export (when status is 'finished'):"
    echo "  curl --header \"PRIVATE-TOKEN: your-token\" \\"
    echo "       --output \"project-export.tar.gz\" \\"
    echo "       http://localhost:8080/api/v4/projects/PROJECT_ID/export/download"
    echo ""
    echo "  # Move to exports directory:"
    echo "  mv project-export.tar.gz $SCRIPT_DIR/gitlab/exports/"
    echo ""
}

# Export OwnCloud data
export_owncloud() {
    echo -e "${YELLOW}Exporting OwnCloud data...${NC}"

    # Find OwnCloud container (could have different names)
    local owncloud_container=$(docker ps --format '{{.Names}}' | grep -i owncloud | grep -v mcp | head -1)

    if [ -z "$owncloud_container" ]; then
        echo -e "${RED}OwnCloud container not running. Start it first with: cd owncloud && docker-compose up -d${NC}"
        return 1
    fi

    echo "Found OwnCloud container: $owncloud_container"
    mkdir -p owncloud/owncloud_data

    echo "Copying data from OwnCloud container..."

    # Try to copy admin data
    if docker cp "$owncloud_container":/mnt/data/admin owncloud/owncloud_data/ 2>/dev/null; then
        echo -e "${GREEN}✓ OwnCloud data exported to: owncloud/owncloud_data/${NC}"
    else
        echo -e "${YELLOW}Trying alternative path...${NC}"
        # Check what paths exist in the container
        echo "Available paths in container:"
        docker exec "$owncloud_container" ls -la /mnt/data/ 2>/dev/null || true

        echo ""
        echo "Try manually with:"
        echo "  docker exec $owncloud_container ls -la /mnt/data/"
        echo "  docker cp $owncloud_container:/mnt/data/admin ./owncloud/owncloud_data/"
    fi

    echo ""
}

# Export RocketChat data
export_rocketchat() {
    echo -e "${YELLOW}Exporting RocketChat data...${NC}"

    # Find MongoDB container (could have different names)
    local mongo_container=$(docker ps --format '{{.Names}}' | grep -i mongo | grep -v mcp | head -1)

    if [ -z "$mongo_container" ]; then
        echo -e "${RED}RocketChat MongoDB container not running.${NC}"
        echo "Start RocketChat first with: cd rocketchat && docker-compose up -d"
        return 1
    fi

    echo "Found MongoDB container: $mongo_container"
    mkdir -p rocketchat

    echo "Exporting MongoDB database..."

    # Export database
    if docker exec "$mongo_container" mongodump \
        --db=rocketchat \
        --archive=/tmp/rocketchat.dump 2>/dev/null; then

        # Copy from container
        docker cp "$mongo_container":/tmp/rocketchat.dump ./rocketchat/rocketchat.dump

        echo -e "${GREEN}✓ RocketChat data exported to: rocketchat/rocketchat.dump${NC}"
    else
        echo -e "${RED}Failed to export RocketChat database.${NC}"
        echo "Try manually with:"
        echo "  docker exec $mongo_container mongodump --db=rocketchat --archive=/tmp/rocketchat.dump"
        echo "  docker cp $mongo_container:/tmp/rocketchat.dump ./rocketchat/"
    fi

    echo ""
}

# Export Plane data
export_plane() {
    echo -e "${YELLOW}Exporting Plane data...${NC}"

    local postgres_container=$(docker ps --format '{{.Names}}' | grep -i plane | grep -i postgres | head -1)

    if [ -z "$postgres_container" ]; then
        echo -e "${RED}Plane PostgreSQL container not running.${NC}"
        return 1
    fi

    echo "Found PostgreSQL container: $postgres_container"
    mkdir -p plane

    echo "Exporting PostgreSQL database..."

    if docker exec "$postgres_container" pg_dump -U plane plane > ./plane/plane.sql 2>/dev/null; then
        echo -e "${GREEN}✓ Plane data exported to: plane/plane.sql${NC}"
    else
        echo -e "${RED}Failed to export Plane database.${NC}"
        echo "Try manually with:"
        echo "  docker exec $postgres_container pg_dump -U plane plane > ./plane/plane.sql"
    fi

    echo ""
}

# Main export logic
echo "Detecting running containers..."
echo ""

# Check what's running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -v "NAMES"

echo ""
echo "Starting exports..."
echo ""

# Export each service
export_gitlab
export_owncloud
export_rocketchat
export_plane

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Export Summary${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check what was exported
if [ -d "gitlab/exports" ] && [ "$(ls -A gitlab/exports 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ GitLab exports: $(ls gitlab/exports | wc -l) file(s)${NC}"
else
    echo -e "${YELLOW}○ GitLab exports: No files (follow instructions above)${NC}"
fi

if [ -d "owncloud/owncloud_data" ] && [ "$(ls -A owncloud/owncloud_data 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ OwnCloud data: Exported${NC}"
else
    echo -e "${YELLOW}○ OwnCloud data: Not exported${NC}"
fi

if [ -f "rocketchat/rocketchat.dump" ]; then
    echo -e "${GREEN}✓ RocketChat dump: $(du -h rocketchat/rocketchat.dump | cut -f1)${NC}"
else
    echo -e "${YELLOW}○ RocketChat dump: Not exported${NC}"
fi

if [ -f "plane/plane.sql" ]; then
    echo -e "${GREEN}✓ Plane dump: $(du -h plane/plane.sql | cut -f1)${NC}"
else
    echo -e "${YELLOW}○ Plane dump: Not exported${NC}"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Verify exported data is complete"
echo "2. Run: ./build_all_images.sh"
echo "3. Update docker-compose.yml files to use custom images"
echo "4. Test with: docker-compose down -v && docker-compose up -d"
echo ""
