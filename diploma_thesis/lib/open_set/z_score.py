import torch

def z_score(train_emb: torch.Tensor, val_emb: torch.Tensor, test_emb: torch.Tensor, imp_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    mean = train_emb.mean(dim = 0)
    std = train_emb.std(dim = 0).clamp_min(1e-8)

    train_emb = (train_emb - mean) / std
    val_emb = (val_emb - mean) / std
    test_emb = (test_emb - mean) / std
    imp_emb = (imp_emb - mean) / std

    return train_emb, val_emb, test_emb, imp_emb