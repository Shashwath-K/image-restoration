"""
services/base.py

Shared helpers for all services.
"""

import torch

def get_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"
