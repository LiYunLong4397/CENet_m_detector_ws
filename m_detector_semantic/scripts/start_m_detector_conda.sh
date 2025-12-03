cat > ~/start_m_detector_conda.sh << 'EOF'
#!/bin/bash

echo "=========================================="
echo "  M-Detector with Conda Environment"
echo "=========================================="

# 1. 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠ Not in conda environment"
    echo "Please activate your conda environment first:"
    echo "  conda activate your_env_name"
    exit 1
fi

echo -e "\n[1/5] Conda Environment"
echo "  Active env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"

# 2. 检查 PyTorch
echo -e "\n[2/5] Checking PyTorch..."
python << 'PYEOF'
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    torch_available = True
except ImportError:
    print("✗ PyTorch not available")
    print("  Install with: conda install pytorch torchvision cpuonly -c pytorch")
    torch_available = False
    exit(1)
PYEOF

if [ $?  -ne 0 ]; then
    read -p "Continue without PyTorch (dummy mode)? [y/N]: " choice
    if [ "$choice" != "y" ]; then
        exit 1
    fi
fi

# 3. 测试 CENet
echo -e "\n[3/5] Testing CENet..."
cd ~/CENet
python -c "
import sys
sys.path.insert(0, '.')
import cenet_inference
success = cenet_inference.init_model(
    'checkpoints/model_best.pth',
    'config/semantic-kitti.yaml'
)
print('✓ CENet OK' if success else '✗ CENet failed')
" 2>&1 | grep -E "(✓|✗|PyTorch|Device)"

# 4. Source ROS
echo -e "\n[4/5] Setting up ROS..."
source /opt/ros/noetic/setup.bash
source ~/CENet_m_detector_ws/devel/setup.bash
echo "✓ ROS environment loaded"

# 5. Launch
echo -e "\n[5/5] Launching M-Detector..."
echo ""
echo "=========================================="
echo "  Starting..."
echo "=========================================="
echo ""

roslaunch m_detector_semantic dynfilter_semantic.launch use_semantic:=true rviz:=true

EOF


