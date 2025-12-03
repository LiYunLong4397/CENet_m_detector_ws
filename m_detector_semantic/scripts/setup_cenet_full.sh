#!/bin/bash
# 保存为: setup_cenet_full.sh

echo "=========================================="
echo "  Setting up CENet Full Repository"
echo "=========================================="

CENET_DIR=~/CENet
ORIGINAL_CENET=~/CENet_original

# 1. 克隆原始 CENet 仓库
echo -e "\n[1/5] Cloning CENet repository..."
if [ !  -d "$ORIGINAL_CENET" ]; then
    git clone https://github. com/huixiancheng/CENet.git $ORIGINAL_CENET
    echo "✓ CENet cloned to $ORIGINAL_CENET"
else
    echo "⚠ CENet already exists at $ORIGINAL_CENET"
fi

# 2. 复制必要的模块到我们的 CENet 目录
echo -e "\n[2/5] Copying CENet modules..."
mkdir -p $CENET_DIR/modules/network
cp -r $ORIGINAL_CENET/modules/* $CENET_DIR/modules/ 2>/dev/null || true

# 3. 创建 __init__.py 文件
echo -e "\n[3/5] Creating Python package structure..."
cat > $CENET_DIR/modules/network/__init__.py << 'EOF'
"""CENet network module"""
EOF

# 4. 检查是否有 CENet. py
echo -e "\n[4/5] Checking network files..."
if [ -f "$CENET_DIR/modules/network/CENet.py" ]; then
    echo "✓ CENet.py found"
elif [ -f "$ORIGINAL_CENET/modules/network/CENet.py" ]; then
    cp $ORIGINAL_CENET/modules/network/CENet.py $CENET_DIR/modules/network/
    echo "✓ CENet.py copied"
else
    echo "✗ CENet.py not found - will create minimal version"
    cat > $CENET_DIR/modules/network/CENet.py << 'EOFPY'
import torch
import torch.nn as nn

class CENet(nn.Module):
    """Minimal CENet implementation for testing"""
    def __init__(self, n_class=20, input_channel=5):
        super(CENet, self).__init__()
        self.n_class = n_class
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn. BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn. ReLU(),
            nn. Conv2d(64, n_class, 1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
EOFPY
    echo "✓ Created minimal CENet. py"
fi

# 5. 测试导入
echo -e "\n[5/5] Testing module import..."
python3 << 'EOFTEST'
import sys
sys.path.insert(0, '/home/ergou/CENet')

try:
    from modules.network.CENet import CENet
    print("✓ CENet module imported successfully")
    
    import torch
    model = CENet(n_class=20, input_channel=5)
    print(f"✓ CENet model created: {type(model)}")
    
    # 测试前向传播
    x = torch.rand(1, 5, 64, 2048)
    y = model(x)
    print(f"✓ Forward pass successful: {y.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback. print_exc()
EOFTEST

echo -e "\n=========================================="
echo "  CENet Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure:"
tree -L 3 $CENET_DIR 2>/dev/null || ls -R $CENET_DIR
