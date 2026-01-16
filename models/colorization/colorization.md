# models/colorization

This module implements **automatic image colorization** — converting grayscale
images into visually plausible color images using deep learning.

Recommended models:
- **DeOldify** — GAN-based colorization and restoration  
  https://github.com/jantic/DeOldify
- **ChromaGAN** — adversarial learning for realistic chroma prediction  
  https://github.com/pvitoria/ChromaGAN

This module provides:
- `model.py`   : Model wrapper and inference logic  
- `utils.py`   : Image I/O and color space utilities  
- `infer.py`   : Command-line interface for colorization  
- `config.yaml`: Runtime configuration  
- `weights/`   : Store pretrained model weights here  

A deterministic **OpenCV colorization fallback** is included to ensure the
pipeline remains operational even without deep models.
