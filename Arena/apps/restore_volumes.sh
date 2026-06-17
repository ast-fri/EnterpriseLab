#!/bin/bash
set -e

# Script to restore Docker volumes from backups
# This runs automatically when users clone your repo and start services

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Restore Docker Volumes${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if backups directory exists
if [ ! -d "volume_backups" ]; then
    echo -e "${RED}No volume_backups directory found!${NC}"
    echo "This means either:"
    echo "  1. You're the maintainer and need to run ./backup_volumes.sh first"
    echo "  2. The backups weren't committed to the repo"
    exit 1
fi

# Check if any backups exist
if [ -z "$(ls -A volume_backups/*.tar.gz 2>/dev/null)" ]; then
    echo -e "${YELLOW}No backup files found in volume_backups/${NC}"
    echo "Skipping volume restore."
    exit 0
fi

restore_volume() {
    local volume_name=$1
    local backup_file=$2
    local service_name=$3

    if [ ! -f "volume_backups/$backup_file" ]; then
        echo -e "${YELLOW}○ $service_name backup not found, skipping...${NC}"
        return
    fi

    # Check if volume already has data
    if docker volume inspect "$volume_name" >/dev/null 2>&1; then
        # Check if volume is empty
        local has_data=$(docker run --rm -v "$volume_name":/data alpine sh -c "ls -A /data | wc -l")
        if [ "$has_data" -gt "0" ]; then
            echo -e "${YELLOW}○ $service_name volume already has data, skipping...${NC}"
            return
        fi
    else
        # Create volume if it doesn't exist
        docker volume create "$volume_name" >/dev/null
    fi

    echo -e "${YELLOW}Restoring $service_name...${NC}"

    # Restore from backup
    docker run --rm \
        -v "$volume_name":/data \
        -v "$(pwd)/volume_backups":/backup \
        alpine tar xzf "/backup/$backup_file" -C /

    echo -e "${GREEN}✓ $service_name restored${NC}"
}

echo "This will restore pre-filled data into Docker volumes."
echo ""

# Restore OwnCloud data
restore_volume "owncloud_owncloud_files" "owncloud-data.tar.gz" "OwnCloud Files"

# Restore OwnCloud database
restore_volume "owncloud_db_data" "owncloud-db.tar.gz" "OwnCloud Database"

# Restore RocketChat MongoDB
restore_volume "rocket-chat_mongodb_data" "rocketchat-mongodb.tar.gz" "RocketChat MongoDB"

# Restore Plane database
restore_volume "plane_pgdata" "plane-db.tar.gz" "Plane Database"

# Restore Plane Redis
restore_volume "plane_redisdata" "plane-redis.tar.gz" "Plane Redis"

# Restore Plane uploads
restore_volume "plane_uploads" "plane-uploads.tar.gz" "Plane Uploads"

# Restore Dolibarr database
restore_volume "dolibarr_dolibarr_db_data" "dolibarr-db.tar.gz" "Dolibarr Database"

# Restore Dolibarr documents
restore_volume "dolibarr_dolibarr_documents" "dolibarr-documents.tar.gz" "Dolibarr Documents"

# Restore GitLab config
restore_volume "gitlab_gitlab_config" "gitlab-config.tar.gz" "GitLab Config"

# Restore GitLab data (if backup exists)
if [ -f "volume_backups/gitlab-data.tar.gz" ]; then
    restore_volume "gitlab_gitlab_data" "gitlab-data.tar.gz" "GitLab Data"
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Volume Restore Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Your services now have pre-filled data!"
echo "You can start them with: ./start_all_servers.sh"
echo ""
