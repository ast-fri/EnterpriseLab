#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Building EnterpriseLab Docker Images${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Function to build an image
build_image() {
    local service_name=$1
    local image_name=$2
    local dir_path=$3

    echo -e "${YELLOW}Building ${service_name}...${NC}"

    if [ ! -d "$dir_path" ]; then
        echo -e "${RED}Directory $dir_path not found. Skipping ${service_name}.${NC}"
        return 1
    fi

    if [ ! -f "$dir_path/Dockerfile" ]; then
        echo -e "${YELLOW}No Dockerfile found for ${service_name}. Skipping.${NC}"
        return 1
    fi

    cd "$dir_path"

    if docker build -t "$image_name" .; then
        echo -e "${GREEN}✓ ${service_name} built successfully!${NC}"
        cd - > /dev/null
        return 0
    else
        echo -e "${RED}✗ Failed to build ${service_name}${NC}"
        cd - > /dev/null
        return 1
    fi
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Track build results
declare -a BUILT_IMAGES
declare -a FAILED_IMAGES

# Build GitLab
if build_image "GitLab" "enterpriselab/gitlab:latest" "gitlab"; then
    BUILT_IMAGES+=("GitLab")
else
    FAILED_IMAGES+=("GitLab")
fi

echo ""

# Build OwnCloud
if build_image "OwnCloud" "enterpriselab/owncloud:latest" "owncloud"; then
    BUILT_IMAGES+=("OwnCloud")
else
    FAILED_IMAGES+=("OwnCloud")
fi

echo ""

# Build RocketChat (if Dockerfile exists)
if build_image "RocketChat" "enterpriselab/rocketchat:latest" "rocketchat"; then
    BUILT_IMAGES+=("RocketChat")
else
    FAILED_IMAGES+=("RocketChat")
fi

echo ""

# Build Plane (if Dockerfile exists)
if build_image "Plane" "enterpriselab/plane:latest" "plane"; then
    BUILT_IMAGES+=("Plane")
else
    FAILED_IMAGES+=("Plane")
fi

echo ""

# Build Dolibarr (if Dockerfile exists)
if build_image "Dolibarr" "enterpriselab/dolibarr:latest" "dolibarr"; then
    BUILT_IMAGES+=("Dolibarr")
else
    FAILED_IMAGES+=("Dolibarr")
fi

echo ""

# Build Frappe (if Dockerfile exists)
if build_image "Frappe" "enterpriselab/frappe:latest" "frappe"; then
    BUILT_IMAGES+=("Frappe")
else
    FAILED_IMAGES+=("Frappe")
fi

echo ""

# Build Zammad (if Dockerfile exists)
if build_image "Zammad" "enterpriselab/zammad:latest" "zammad"; then
    BUILT_IMAGES+=("Zammad")
else
    FAILED_IMAGES+=("Zammad")
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Build Summary${NC}"
echo -e "${GREEN}======================================${NC}"

if [ ${#BUILT_IMAGES[@]} -gt 0 ]; then
    echo -e "${GREEN}Successfully built (${#BUILT_IMAGES[@]}):${NC}"
    for img in "${BUILT_IMAGES[@]}"; do
        echo -e "  ${GREEN}✓${NC} $img"
    done
fi

echo ""

if [ ${#FAILED_IMAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Skipped/Failed (${#FAILED_IMAGES[@]}):${NC}"
    for img in "${FAILED_IMAGES[@]}"; do
        echo -e "  ${YELLOW}○${NC} $img"
    done
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Available EnterpriseLab Images:${NC}"
echo -e "${GREEN}======================================${NC}"
docker images | grep -E "REPOSITORY|enterpriselab" || echo "No enterpriselab images found"

echo ""
echo -e "${GREEN}Done!${NC}"
