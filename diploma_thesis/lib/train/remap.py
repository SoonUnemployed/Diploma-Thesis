import numpy as np 
from pathlib import Path 
import json

def remap(y: np.array, map_path: Path) -> np.array:
    
    load_path = map_path / ("label_mapping.json")
    
    with open (load_path) as f:
        label_map = json.load(f)

    label_map = {int(k): int(v) for k, v in label_map.items()}
    
    y_mapped = np.array([label_map[int(user)] for user in y])

    return y_mapped