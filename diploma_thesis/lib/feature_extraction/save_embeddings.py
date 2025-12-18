import torch
import numpy as np
from pathlib import Path

def save_embeddings(embeddings: torch.Tensor , group: list, y: np.array, output_path: Path, split: str):
    
    save_path = output_path / (f"{split}_embeddings.pt")
    
    torch.save(
        
        {
            "embeddings": embeddings,
            "subjects": y,
            "sessions": group,
        },

        save_path
    )

    return