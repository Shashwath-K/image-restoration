
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.inpainting.unet import UNet

def generate_weights():
    print("Generating placeholder weights for Inpainting UNet...")
    
    # Initialize model (random weights)
    net = UNet(n_channels=4, n_classes=3, bilinear=True)
    
    output_dir = "models/inpainting/weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "inpainting.pth")
    
    torch.save(net.state_dict(), output_path)
    print(f"Saved initialized (untrained) weights to {output_path}")

if __name__ == "__main__":
    generate_weights()
