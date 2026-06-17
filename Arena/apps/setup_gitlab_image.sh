#!/bin/bash
set -e

# Script to build GitLab custom image with pre-filled data
# This is for the custom image approach (hybrid setup)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Setup GitLab Custom Image${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/gitlab"

# Check if exports exist
if [ ! -d "exports" ] || [ -z "$(ls -A exports/*.tar.gz 2>/dev/null)" ]; then
    echo -e "${YELLOW}No GitLab project exports found!${NC}"
    echo ""
    echo "To create a GitLab custom image with your projects:"
    echo ""
    echo "1. Export your GitLab projects:"
    echo "   - Go to http://localhost:8080"
    echo "   - Login as root"
    echo "   - For each project: Settings → General → Advanced → Export project"
    echo "   - Download .tar.gz files"
    echo ""
    echo "2. Place exports in: $SCRIPT_DIR/gitlab/exports/"
    echo ""
    echo "3. Run this script again"
    echo ""
    echo -e "${YELLOW}For now, GitLab will use the standard image.${NC}"
    exit 0
fi

echo "Found GitLab exports:"
ls -lh exports/*.tar.gz
echo ""

echo "Building GitLab custom image..."
echo "⚠️  This takes 15-20 minutes as it starts GitLab and imports projects."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Build the image
docker build -t enterpriselab/gitlab:latest .

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}GitLab Image Built Successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "To use the custom image:"
echo "1. Stop GitLab: cd gitlab && docker compose down"
echo "2. Edit docker-compose.yml:"
echo "   Change: image: gitlab/gitlab-ce:latest"
echo "   To:     image: enterpriselab/gitlab:latest"
echo "3. Restart: docker compose up -d"
echo ""
