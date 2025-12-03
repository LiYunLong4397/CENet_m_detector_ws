#!/usr/bin/env python3
import torch
import numpy as np
import yaml
import sys
import os

# Global model instance
_model = None
_config = None
_device = None

def init_model(model_path, config_path):
    """Initialize CENet model"""
    global _model, _config, _device
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
        
        # Set device
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {_device}")
        
        # Try to import CENet
        try:
            # Add CENet path to system path
            cenet_path = os.path. dirname(os.path.dirname(model_path))
            if cenet_path not in sys. path:
                sys.path. insert(0, cenet_path)
            
            from modules.network. CENet import CENet
            
            # Create model
            _model = CENet(
                n_class=_config. get('model', {}).get('n_class', 20),
                input_channel=_config.get('model', {}).get('input_channel', 5)
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=_device)
            if 'state_dict' in checkpoint:
                _model.load_state_dict(checkpoint['state_dict'])
            else:
                _model.load_state_dict(checkpoint)
            
            _model.to(_device)
            _model.eval()
            
            print(f"CENet model loaded successfully from {model_path}")
            return True
            
        except ImportError as e:
            print(f"Warning: Could not import CENet: {e}")
            print("Using dummy semantic segmentation")
            _model = None
            return True
            
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def points_to_range_image(points):
    """Convert point cloud to range image"""
    # Get sensor configuration
    sensor_cfg = _config.get('dataset', {}).get('sensor', {})
    fov_up = sensor_cfg.get('fov_up', 3.0) / 180.0 * np.pi
    fov_down = sensor_cfg.get('fov_down', -25.0) / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)
    
    img_prop = sensor_cfg.get('img_prop', {})
    proj_H = img_prop.get('height', 64)
    proj_W = img_prop.get('width', 2048)
    
    # Calculate depth
    xyz = points[:, :3]
    depth = np.linalg. norm(xyz, 2, axis=1)
    
    # Calculate pitch and yaw
    pitch = np.arcsin(xyz[:, 2] / (depth + 1e-8))
    yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
    
    # Project to range image
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov
    
    proj_x *= proj_W
    proj_y *= proj_H
    
    proj_x = np.floor(proj_x). astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)
    
    # Clip to valid range
    proj_x = np.clip(proj_x, 0, proj_W - 1)
    proj_y = np.clip(proj_y, 0, proj_H - 1)
    
    # Create range image
    range_image = np.zeros((proj_H, proj_W, 5), dtype=np.float32)
    
    # Fill range image (keep farthest point for each pixel)
    order = np.argsort(depth)[::-1]
    for idx in order:
        px, py = proj_x[idx], proj_y[idx]
        range_image[py, px, 0] = xyz[idx, 0]
        range_image[py, px, 1] = xyz[idx, 1]
        range_image[py, px, 2] = xyz[idx, 2]
        range_image[py, px, 3] = points[idx, 3] if points. shape[1] > 3 else 0.0
        range_image[py, px, 4] = depth[idx]
    
    return range_image, proj_x, proj_y

def dummy_inference(points):
    """Dummy inference when CENet is not available"""
    n_points = points.shape[0]
    # Simple heuristic: label based on height
    labels = np.zeros(n_points, dtype=np.int32)
    confidences = np.ones(n_points, dtype=np.float32) * 0.5
    
    for i in range(n_points):
        z = points[i, 2]
        if z < -1.5:
            labels[i] = 9  # ROAD
        elif z > 2.0:
            labels[i] = 13  # BUILDING
        else:
            labels[i] = 0  # UNLABELED
    
    return labels, confidences

def inference(points):
    """Perform semantic segmentation inference"""
    global _model, _device
    
    if _model is None:
        # Use dummy inference
        return dummy_inference(points)
    
    try:
        # Convert to range image
        range_image, proj_x, proj_y = points_to_range_image(points)
        
        # Convert to tensor
        range_tensor = torch.from_numpy(range_image).permute(2, 0, 1).unsqueeze(0)
        range_tensor = range_tensor.to(_device)
        
        # Inference
        with torch.no_grad():
            output = _model(range_tensor)
            pred_prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1). squeeze(0)
            conf = pred_prob. max(dim=1)[0].squeeze(0)
        
        # Extract point labels
        pred_np = pred.cpu().numpy()
        conf_np = conf.cpu().numpy()
        
        point_labels = pred_np[proj_y, proj_x]
        point_confidences = conf_np[proj_y, proj_x]
        
        return point_labels. astype(np.int32), point_confidences.astype(np.float32)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return dummy_inference(points)

if __name__ == "__main__":
    # Test
    print("CENet inference module loaded")
