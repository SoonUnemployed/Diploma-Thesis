import torch 

def separate(emb: torch.Tensor, y: torch.Tensor, grp: torch.Tensor, temp: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    known = torch.tensor(list(temp.keys()))

    mask = torch.isin(y, known)

    emb_kn = emb[mask]
    y_kn = y[mask]
    grp_kn = grp[mask]
    
    mask = ~mask

    emb_unkn = emb[mask]
    y_unkn = y[mask]
    grp_unkn = grp[mask]

    return emb_kn, y_kn, grp_kn, emb_unkn, y_unkn, grp_unkn