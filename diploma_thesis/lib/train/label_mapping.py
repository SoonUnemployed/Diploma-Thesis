import pandas as pd
from pathlib import Path
import json

def label_mapping(df: pd.DataFrame, output_path: Path):
    
    label_map = {}
        
    df_split = df[df["split"] == "train"]
    users = df_split["user"].values.tolist()
    users = list(set(users))

    users = [str(i) for i in users]

    for i, u in enumerate(users):
        label_map[u] = i

    label_map = {int(k): int(v) for k, v in label_map.items()}
    save_path = output_path / ("label_mapping.json")

    with open(save_path, "w") as f:
        json.dump(label_map, f, indent = 4)
        
    return 