"""
Reproducible seeding utility across Python, NumPy, and PyTorch.
"""
import random
import os
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int = 42) -> None:
    """Set random seed everywhere for strict reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
