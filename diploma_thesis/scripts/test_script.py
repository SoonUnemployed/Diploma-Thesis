from pathlib import Path
import json 
import numpy as np

load_path = Path("/Users/hrakol/projects/diploma_thesis/results/xgboost_closed_set_metrics.npz")

data = np.load(load_path, allow_pickle = True)

print(data["accuracy"])
print(data["confusion_matrix"])
print(data["number_of_users"])
print(data["chance"])