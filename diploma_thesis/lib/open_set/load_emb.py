import torch 
from pathlib import Path
import numpy as np

def load_embeddings(path: Path) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    
    data = torch.load(path, weights_only = False)
    
    return (
        data["embeddings"],
        data["subjects"],
        data["sessions"]
    )
