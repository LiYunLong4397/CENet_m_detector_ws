#!/bin/bash

echo "Setting up CENet for M-Detector integration..."

# Check if CENet repository exists
if [ ! -d "$HOME/CENet" ]; then
    echo "Cloning CENet repository..."
    cd $HOME
    git clone https://github.com/huixiancheng/CENet.git
    cd CENet
else
    echo "CENet repository already exists at $HOME/CENet"
    cd $HOME/CENet
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch. org/whl/cu118
pip3 install numpy pyyaml scipy tqdm

# Download pretrained model (if available)
echo "Setting up model directory..."
mkdir -p checkpoints

echo "CENet setup complete!"
echo "Please download pretrained model to: $HOME/CENet/checkpoints/model_best.pth"
echo "You can train your own model or use pretrained weights from the CENet repository"