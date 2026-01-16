
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.colorization.eccv16 import ColorNet

def generate_weights():
    print("Generating placeholder weights for Colorization (ResNet-UNet)...")
    
    # Initialize model (random weights)
    # Note: This will download standard resnet18 weights if available/internet access works,
    # or just initialize if not. We set pretrained=False to be safe/standalone, 
    # but the class definition has an option.
    net = ColorNet(pretrained=False)
    
    output_dir = "models/colorization/weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "colorizer.pth")
    
    torch.save(net.state_dict(), output_path)
    print(f"Saved initialized (untrained) weights to {output_path}")
    print("WARNING: These are untrained weights. The model will run but produce garbage output until real weights are loaded.")

if __name__ == "__main__":
    generate_weights()
