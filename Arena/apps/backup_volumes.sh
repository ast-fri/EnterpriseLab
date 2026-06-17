#!/bin/bash
set -e

# Script to backup Docker volumes with data
# Run this ONCE to create volume snapshots from your running containers

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Backup Docker Volumes${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create backups directory
mkdir -p volume_backups

backup_volume() {
    local volume_name=$1
    local backup_file=$2
    local service_name=$3

    if docker volume inspect "$volume_name" >/dev/null 2>&1; then
        echo -e "${YELLOW}Backing up $service_name...${NC}"

        # Create backup using alpine container
        docker run --rm \
            -v "$volume_name":/data \
            -v "$(pwd)/volume_backups":/backup \
            alpine tar czf "/backup/$backup_file" -C / data

        local size=$(du -h "volume_backups/$backup_file" | cut -f1)
        echo -e "${GREEN}✓ $service_name backed up ($size)${NC}"
        echo ""
    else
        echo -e "${RED}✗ Volume $volume_name not found. Is $service_name running?${NC}"
        echo ""
    fi
}

echo "This will backup Docker volumes from your running services."
echo "Make sure all services are running before backing up!"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Backup OwnCloud data
backup_volume "owncloud_owncloud_files" "owncloud-data.tar.gz" "OwnCloud"

# Backup OwnCloud database
backup_volume "owncloud_db_data" "owncloud-db.tar.gz" "OwnCloud Database"

# Backup RocketChat MongoDB (if using volumes)
backup_volume "rocket-chat_mongodb_data" "rocketchat-mongodb.tar.gz" "RocketChat MongoDB"

# Backup Plane database
backup_volume "plane_pgdata" "plane-db.tar.gz" "Plane Database"

# Backup Plane Redis
backup_volume "plane_redisdata" "plane-redis.tar.gz" "Plane Redis"

# Backup Plane uploads
backup_volume "plane_uploads" "plane-uploads.tar.gz" "Plane Uploads"

# Backup Dolibarr database
backup_volume "dolibarr_dolibarr_db_data" "dolibarr-db.tar.gz" "Dolibarr Database"

# Backup Dolibarr documents
backup_volume "dolibarr_dolibarr_documents" "dolibarr-documents.tar.gz" "Dolibarr Documents"

# Backup GitLab config (optional)
backup_volume "gitlab_gitlab_config" "gitlab-config.tar.gz" "GitLab Config"

# Backup GitLab data (optional - might be large!)
if [ "$1" == "--with-gitlab-data" ]; then
    backup_volume "gitlab_gitlab_data" "gitlab-data.tar.gz" "GitLab Data"
fi

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Backup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Backups saved to: $(pwd)/volume_backups/"
echo ""
ls -lh volume_backups/
echo ""
echo -e "${YELLOW}Important: Commit these backups to your git repo!${NC}"
echo ""
echo "  git add volume_backups/"
echo "  git commit -m 'Add volume backups with pre-filled data'"
echo "  git push"
echo ""
echo "Now users who clone your repo will have pre-filled data!"
