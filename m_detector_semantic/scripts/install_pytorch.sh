#!/bin/bash
# 保存为: install_pytorch.sh

echo "=========================================="
echo "  Installing PyTorch for CENet"
echo "=========================================="

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
    
    # 检查 CUDA 版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,. *//')
        echo "✓ CUDA version: $CUDA_VERSION"
        
        # 根据 CUDA 版本选择
        if [[ "$CUDA_VERSION" =~ ^11\.[0-7] ]]; then
            echo "Installing PyTorch with CUDA 11.8..."
            pip3 install torch torchvision torchaudio --index-url https://download. pytorch.org/whl/cu118
        elif [[ "$CUDA_VERSION" =~ ^12 ]]; then
            echo "Installing PyTorch with CUDA 12.1..."
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            echo "Installing PyTorch with CUDA 11.8 (default)..."
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        echo "CUDA toolkit not found, installing CPU version..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo -e "\n⚠ No NVIDIA GPU detected"
    echo "Installing PyTorch CPU version..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 验证安装
echo -e "\n=========================================="
echo "  Verifying PyTorch Installation"
echo "=========================================="

python3 << 'EOF'
import torch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda. is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch. version.cuda}")
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("  Running on CPU")

# 简单测试
x = torch.rand(5, 3)
print(f"\n✓ PyTorch working correctly")
print(f"  Test tensor shape: {x.shape}")
EOF

echo -e "\n✓ PyTorch installation complete!"
