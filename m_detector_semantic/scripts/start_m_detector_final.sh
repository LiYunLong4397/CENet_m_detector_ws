
#!/bin/bash

echo "=========================================="
echo "  M-Detector with CENet (GPU Accelerated)"
echo "=========================================="

# 1. 激活 conda 环境
echo -e "\n[1/6] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cenet_env

if [ "$CONDA_DEFAULT_ENV" != "cenet_env" ]; then
    echo "✗ Failed to activate cenet_env"
    exit 1
fi
echo "✓ Environment: $CONDA_DEFAULT_ENV"

# 2. 验证 PyTorch + GPU
echo -e "\n[2/6] Checking PyTorch..."
python << 'PYEOF'
import torch
print(f"✓ PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ Running on CPU")
PYEOF

# 3. 检查模型
echo -e "\n[3/6] Checking model..."
if [ -f ~/CENet/checkpoints/model_best.pth ]; then
    SIZE=$(du -h ~/CENet/checkpoints/model_best.pth | cut -f1)
    echo "✓ Model found: $SIZE"
else
    echo "⚠ No pretrained model (using random initialization)"
fi

# 4. 测试 CENet
echo -e "\n[4/6] Testing CENet inference..."
cd ~/CENet
python cenet_inference.py 2>&1 | grep -E "(PyTorch|Device|GPU|Model|SUCCESS)" | head -8

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "✗ CENet test failed"
    exit 1
fi

# 5. 设置 ROS 环境
echo -e "\n[5/6] Setting up ROS..."
export PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source /opt/ros/noetic/setup.bash
source ~/CENet_m_detector_ws/devel/setup.bash
echo "✓ ROS environment ready"

# 6. 启动
echo -e "\n[6/6] Launching M-Detector..."
echo ""
echo "=========================================="
echo "  System Starting..."
echo "=========================================="
echo ""
echo "📊 Monitoring topics:"
echo "  /m_detector/point_out     - Dynamic points with semantics"
echo "  /m_detector/frame_out     - Clustered dynamic objects"
echo "  /m_detector/std_points    - Static points"
echo "  /m_detector/semantic_colored - Semantic visualization"
echo ""
echo "🎮 Control:"
echo "  Ctrl+C to stop"
echo ""

cd ~/CENet_m_detector_ws
roslaunch m_detector_semantic dynfilter_semantic.launch 

EOF

chmod +x ~/start_m_detector_final.sh
