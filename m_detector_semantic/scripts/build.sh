#!/bin/bash

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  M-Detector Semantic Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Check Python3
if !  command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found! ${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 found${NC}"

# Check NumPy
python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}NumPy not found!${NC}"
    echo "Installing NumPy..."
    pip3 install numpy
fi
echo -e "${GREEN}✓ NumPy found${NC}"

# Check PyTorch (optional)
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ PyTorch not found (optional for semantic segmentation)${NC}"
    echo "Install with: pip3 install torch torchvision"
else
    echo -e "${GREEN}✓ PyTorch found${NC}"
fi

# Check ROS
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}⚠ ROS not sourced, sourcing...${NC}"
    if [ -f "/opt/ros/noetic/setup.bash" ]; then
        source /opt/ros/noetic/setup.bash
    elif [ -f "/opt/ros/melodic/setup.bash" ]; then
        source /opt/ros/melodic/setup.bash
    else
        echo -e "${RED}ROS not found!${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ ROS $ROS_DISTRO found${NC}"

# Find catkin workspace root
# Assume structure: <workspace>/src/m_detector_semantic/scripts/build.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CATKIN_WS="$(cd "$SCRIPT_DIR/../../. ." && pwd)"

echo -e "\n${YELLOW}Catkin workspace: $CATKIN_WS${NC}"

# Verify it's a catkin workspace
if [ ! -f "$CATKIN_WS/src/CMakeLists.txt" ]; then
    echo -e "${RED}Error: $CATKIN_WS does not appear to be a catkin workspace${NC}"
    echo -e "${YELLOW}Expected to find: $CATKIN_WS/src/CMakeLists.txt${NC}"
    exit 1
fi

# Navigate to workspace
cd "$CATKIN_WS"
echo -e "${YELLOW}Building in: $(pwd)${NC}"

# Build
echo -e "\n${YELLOW}Building m_detector_semantic...${NC}"
catkin_make -DCMAKE_BUILD_TYPE=Release --pkg m_detector_semantic

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build Successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\n${YELLOW}To use the package, run:${NC}"
    echo -e "  ${GREEN}source $CATKIN_WS/devel/setup.bash${NC}"
    echo -e "  ${GREEN}roslaunch m_detector_semantic dynfilter_semantic.launch${NC}"
    
    # Auto source
    source "$CATKIN_WS/devel/setup.bash"
    echo -e "\n${GREEN}✓ Environment sourced automatically${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}  Build Failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
