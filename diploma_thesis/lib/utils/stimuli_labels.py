from pathlib import Path 
import os 
import pandas as pd
import numpy as np

def load_stim(path: Path) -> dict:
    
    files = [f for f in os.listdir(path) if f.endswith(".txt")]
    stim_seq = {}

    for file in files:
    
        load_path = path / file 
        file = file[:-4]

        with open(load_path, "r") as f:
            labels = list(map(int, f.read().split()))
            
        stim_seq[file] = labels
    
    return stim_seq

def select_seq(stim_seq: dict, split_df: pd.DataFrame, split_type: str) -> dict:

    kept_seq = {}

    temp = split_df[split_df["split"] == split_type]
    sessions = temp["session"].tolist()

    for s in sessions:

        e = stim_seq.get(s)
        kept_seq[s] = e
    
    return kept_seq

def arrange_stim(seq: dict, group: np.array) -> np.array:

    temp = []
    prev = object()

    for i in group:
        if i == prev:
            continue

        e = seq.get(i)
        temp.extend(e)
        prev = i    

    return np.array(temp)
