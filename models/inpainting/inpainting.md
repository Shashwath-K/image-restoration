# models/inpainting

This module implements **image inpainting** — restoration of missing or corrupted
regions in images using deep learning.

Recommended model:
- **LaMa (Large Mask Inpainting)**  
  https://github.com/advimman/lama

This folder provides:
- `model.py`   : Model wrapper and safe fallback logic  
- `utils.py`   : Image IO and mask helpers  
- `infer.py`   : CLI utility for inpainting images  
- `config.yaml`: Runtime configuration  
- `weights/`   : Store pretrained model weights here  

If pretrained weights are not present, the system falls back to a
deterministic OpenCV-based inpainting method so the pipeline remains usable.
