from pathlib import Path
import numpy as np

def load_features(load_path: Path) -> list:

    features = np.load(load_path, allow_pickle = True)

    return features