import torch
from lib.open_set.similarity import cosine_sim

def build_templates(E_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    
    templates = {}

    for subj in torch.unique(y_train):
        subj = int(subj)
        mu = E_train[y_train == subj].mean(0)

        #mu = torch.nn.functional.normalize(mu, dim=0)

        templates[subj] = mu

    return templates


def calibrate_threshold(E_val: torch.Tensor, y_val: torch.Tensor, templates: dict, target_far: float = 0.1) -> dict:
    
    threshold = {}

    for subj, mu in templates.items():

        E_g = E_val[y_val == subj]

        #scores = (E_g * mu).sum(dim = 1)

        scores = torch.Tensor([cosine_sim(e , mu) for e in E_g])

        thres = torch.quantile(scores, target_far).item()
        threshold[subj] = thres

    return threshold
