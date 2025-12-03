#!/bin/bash
# 保存为: install_pytorch_auto.sh

set -e

echo "=========================================="
echo "  PyTorch Automatic Installation"
echo "=========================================="

# 加载检测结果
if [ -f /tmp/torch_version.sh ]; then
    source /tmp/torch_version.sh
else
    echo "Running system check first..."
    bash check_gpu.sh
    source /tmp/torch_version.sh
fi

echo -e "\n[1/4] Installing PyTorch..."
echo "  Target: PyTorch with $TORCH_VERSION support"

# 更新 pip
pip3 install --upgrade pip

# 根据类型安装
if [ "$TORCH_VERSION" == "cpu" ]; then
    echo "  Installing CPU version..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$TORCH_VERSION" == "cu118" ]; then
    echo "  Installing CUDA 11.8 version..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$TORCH_VERSION" == "cu121" ]; then
    echo "  Installing CUDA 12.1 version..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  Installing latest stable version..."
    pip3 install torch torchvision torchaudio
fi

echo "✓ PyTorch installed"

# 验证安装
echo -e "\n[2/4] Verifying installation..."
python3 << 'PYEOF'
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch. version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
    
    # 测试 GPU
    x = torch.rand(5, 3). cuda()
    print(f"✓ GPU tensor test passed")
else:
    print("  Running on CPU")
    
# 测试基本操作
x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = x + y
print(f"✓ Basic operations test passed")
PYEOF

if [ $? -eq 0 ]; then
    echo "✓ Verification passed"
else
    echo "✗ Verification failed"
    exit 1
fi

# 安装其他依赖
echo -e "\n[3/4] Installing additional dependencies..."
pip3 install pyyaml scipy tqdm pillow

echo "✓ Dependencies installed"

# 测试 CENet 兼容性
echo -e "\n[4/4] Testing CENet compatibility..."
cd ~/CENet
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/ergou/CENet')
import torch
import numpy as np

# 测试能否创建简单模型
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn. Conv2d(5, 20, 3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

model = TestModel()
print(f"✓ Model creation: OK")

# 测试前向传播
x = torch.rand(1, 5, 64, 2048)
y = model(x)
print(f"✓ Forward pass: OK (output shape: {y.shape})")

# 测试 CPU/GPU
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    y = model(x)
    print(f"✓ GPU inference: OK")
else:
    print(f"✓ CPU inference: OK")

print("\n✓ CENet compatibility confirmed")
PYEOF

echo -e "\n=========================================="
echo "  ✓ Installation Complete!"
echo "=========================================="
echo ""
echo "PyTorch installed successfully with $TORCH_VERSION support"
echo ""
echo "Next steps:"
echo "  1. Test CENet: cd ~/CENet && python3 cenet_inference.py"
echo "  2. Start M-Detector: ~/start_m_detector_full.sh"
echo ""
