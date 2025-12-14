from braindecode.models import EEGNet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from lib.train.dataset import EEGNet_Dataset
from lib.test.metrics import evaluate_model

def eegnet(X: np.array, y: np.array, device: torch.device) -> EEGNet:

    model = EEGNet(n_chans = X.shape[1], n_outputs = len(np.unique(y)), n_times = X.shape[2], sfreq = 256.0).to(device)

    return model


def train_eegnet(model: EEGNet, X: np.array, y: np.array, X_val: np.array, y_val: np.array, device: torch.device) -> tuple[EEGNet, float]:
    
    #Code without CV
    train_dataset = EEGNet_Dataset(X, y)
    val_dataset = EEGNet_Dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 300
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 25         
    min_delta = 1e-4       #Ignore tiny changes in val loss
    best_state = None

    for epoch in range(epochs): 

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (outputs.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y, in val_loader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item() * X.size(0)
                val_correct += (outputs.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        #Early stopping 

        if val_acc > best_val_acc + 5e-3:  
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


        if best_state is not None:
            model.load_state_dict(best_state)

    return model, val_acc
    '''
    #Code with CV
    X_t = torch.tensor(X, dtype = torch.float32).to(device)
    y_t = torch.tensor(y, dtype = torch.long).to(device)
    X_val_t  = torch.tensor(X_val, dtype = torch.float32).to(device)
    y_val_t   = torch.tensor(y_val, dtype = torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    epochs = 150

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        preds = model(X_val_t).argmax(dim = 1)
        acc = (preds == y_val_t).float().mean().item()
    '''

    return model, acc

def test_eegnet(model: EEGNet, X: np.array, y: np.array, model_path: Path, device: torch.device) -> tuple[dict, np.array]:
    m_path = model_path / ("eegnet.pth")
    test_dataset = EEGNet_Dataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    model.load_state_dict(torch.load(m_path))
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            probs = outputs.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(y_batch.cpu().numpy().astype(int))

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    all_preds = all_probs.argmax(axis = 1).astype(int)

    metrics = evaluate_model(all_preds, all_labels)

    print("\n===== EEGNet Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: \n{v}\n")

    return metrics, all_preds