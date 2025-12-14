import pandas as pd 
import numpy as np
import joblib
import torch
import pickle 
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


from lib.utils.epoch_data import epoch_data

def load_dataset(df: pd.DataFrame,  preprocessed_path: Path, features_path: Path, set_type: str, model: str) -> tuple[np.array, np.array, np.array]:
    
    X = []
    y = []
    groups = []

    if set_type in ["train", "val", "test"]:
        temp = df[df["split"] == set_type]
    else: 
        temp = df[df["split"] != "impostor"]

    for _, row in temp.iterrows():
        
        if model == "eegnet":
            load_path = preprocessed_path / (str(row["session"]) + ".fif")
            epochs = epoch_data(load_path)
            epochs = epochs.get_data()
            '''
            ------Add epoch selector for specific frequencies------
            '''
            X.append(epochs)
            y.append(np.full(len(epochs), row["user"]))
            groups.append(np.full(len(epochs), row["session"]))

        elif model == "xgboost":
            load_path = features_path / (str(row["session"]) + ".npy")
            features = np.load(load_path)
            '''
            ------Add epoch selector for specific frequencies------
            '''
            X.append(features)
            y.append(np.full(features.shape[0], row["user"]))
            groups.append(np.full(features.shape[0], row["session"]))

    X = np.concatenate(X)
    y = np.concatenate(y)
    groups = np.concatenate(groups)

    return X, y, groups

def reshape_dataset(X: np.array, model: str) -> np.array:

    if model == "eegnet":
        X = np.expand_dims(X, axis = -1)

    elif model == "xgboost":
        pass

    return X

def apply_scalar(X: np.array, model: str, set_type: str, path: Path) -> np.array:

    if model == "xgboost":    
        if set_type == "train": 
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
            save_path = path / (f"scaler_{model}.pkl")
            joblib.dump(scaler, save_path)

        else:
            load_path = path / (f"scaler_{model}.pkl")
            scaler = joblib.load(load_path)
            X_norm = scaler.transform(X)
    
    if model == "eegnet":
        if set_type == "train":    
            mean = X.mean(axis = (0, 2), keepdims = True)
            std = X.std(axis = (0, 2), keepdims = True)
            std[std == 0] = 1
            X_norm = (X - mean) / std

            save_path = path / (f"normalization_{model}.pkl")
            pickle.dump({"mean": mean, "std": std}, open(save_path, "wb"))
            

        else:
            load_path = path / (f"normalization_{model}.pkl")
            norm = pickle.load(open(load_path, "rb"))
            mean, std = norm["mean"], norm["std"]
            X_norm = (X - mean) / std

    return X_norm

class EEGNet_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]