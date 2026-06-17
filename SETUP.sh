#!/bin/bash
set -e

# EnterpriseLab Setup Script
# Run this after cloning the repository

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════╗
║                                                   ║
║           🏢  ENTERPRISE LAB SETUP  🏢            ║
║                                                   ║
║        Automated Multi-Service Environment        ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check Docker
echo -e "${BLUE}[1/4] Checking Docker installation...${NC}"
if ! command -v docker &>/dev/null; then
    echo -e "${RED}❌ Docker not found!${NC}"
    echo ""
    echo "Please install Docker first:"
    echo "  https://docs.docker.com/get-docker/"
    exit 1
fi

if docker compose version &>/dev/null; then
    COMPOSE_CMD="docker compose"
elif docker-compose version &>/dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}❌ Docker Compose not found!${NC}"
    echo ""
    echo "Please install Docker Compose:"
    echo "  https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose found!${NC}"
echo ""

# Check for volume backups
echo -e "${BLUE}[2/4] Checking for pre-filled data...${NC}"
if [ -d "Arena/apps/volume_backups" ] && [ -n "$(ls -A Arena/apps/volume_backups/*.tar.gz 2>/dev/null)" ]; then
    echo -e "${GREEN}✅ Pre-filled data found!${NC}"
    echo ""
    echo "Found backups:"
    ls -lh Arena/apps/volume_backups/*.tar.gz | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
    HAS_BACKUPS=true
else
    echo -e "${YELLOW}⚠️  No pre-filled data found${NC}"
    echo ""
    echo "This means services will start empty."
    echo "If you're the maintainer, run: cd Arena/apps && ./backup_volumes.sh"
    echo ""
    HAS_BACKUPS=false
fi

# Pull Docker images
echo -e "${BLUE}[3/4] Pulling Docker images...${NC}"
echo "This may take a few minutes..."
echo ""

cd "$SCRIPT_DIR/Arena/apps"

# Pull images for main services
declare -a SERVICES=("gitlab" "owncloud" "rocketchat" "plane" "dolibarr")

for service in "${SERVICES[@]}"; do
    if [ -d "$service" ] && [ -f "$service/docker-compose.yml" -o -f "$service/docker-compose.yaml" ]; then
        echo -e "${YELLOW}Pulling images for $service...${NC}"
        (cd "$service" && $COMPOSE_CMD pull) || echo "  (Some images may not need pulling)"
    fi
done

echo ""
echo -e "${GREEN}✅ Images pulled!${NC}"
echo ""

# Setup complete
echo -e "${BLUE}[4/4] Final setup...${NC}"

# Create marker file for first run
if [ -f ".volumes_restored" ]; then
    rm .volumes_restored
fi

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""

# Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              SETUP COMPLETE! 🎉                   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$HAS_BACKUPS" = true ]; then
    echo -e "${GREEN}✅ Pre-filled data will be restored automatically${NC}"
    echo ""
fi

echo "To start all services:"
echo -e "${YELLOW}  cd Arena/apps${NC}"
echo -e "${YELLOW}  ./start_all_servers.sh${NC}"
echo ""

if [ "$HAS_BACKUPS" = true ]; then
    echo "On first start, your services will have:"
    echo "  ✅ Pre-filled data from volume backups"
    echo "  ✅ Ready-to-use configurations"
    echo ""
fi

echo "Services will be available at:"
echo "  🦊 GitLab       → http://localhost:8080"
echo "  ☁️  OwnCloud    → http://localhost:8081"
echo "  💬 RocketChat   → http://localhost:3000"
echo "  ✈️  Plane       → http://localhost:3001"
echo "  💼 Dolibarr     → http://localhost:8082"
echo ""

echo "Other commands:"
echo "  ./start_all_servers.sh status  → Check service status"
echo "  ./start_all_servers.sh stop    → Stop all services"
echo ""

read -p "Start services now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}🚀 Starting services...${NC}"
    echo ""
    ./start_all_servers.sh
else
    echo ""
    echo "Run './start_all_servers.sh' when ready!"
fi

echo ""
echo -e "${GREEN}Thank you for using EnterpriseLab! 🏢${NC}"
echo ""
