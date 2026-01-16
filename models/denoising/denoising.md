# models/denoising

This module handles **image denoising and artifact suppression** using
deep learning–based restoration techniques.

Recommended models:
- **DnCNN** – residual learning CNN for Gaussian noise removal  
  https://github.com/cszn/DnCNN
- **NAFNet / Restormer** – modern transformer-based image restoration  
  (can be integrated later for advanced results)

This module provides:
- `model.py`   : Model wrapper and inference logic  
- `utils.py`   : Image I/O and noise utilities  
- `infer.py`   : Command-line interface for denoising  
- `config.yaml`: Runtime configuration  
- `weights/`   : Store pretrained model weights here  

A safe **Gaussian blur fallback** is included so the pipeline remains
functional even without deep models.
