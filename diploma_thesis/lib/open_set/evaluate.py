import torch 
from collections import defaultdict
from lib.open_set.similarity import cosine_sim

def verify_claim(e: torch.Tensor, claimed_id: int, templates: dict, thresholds: dict) -> tuple[bool, float]:
    
    mu = templates[claimed_id]
    score = cosine_sim(e, mu)
    #score = torch.dot(e,mu).item()
    accept = score >= thresholds[claimed_id]

    return accept, score

def eval_known(E_known: torch.Tensor, y_known: torch.Tensor, grp_known: torch.Tensor, templates: dict, thresholds: dict) -> tuple[float, int, int]:
    
    TP = 0
    FN = 0 
    total = len(E_known)

    for i in range(len(y_known)):
        true_id = int(y_known[i])
        accept, _ = verify_claim(E_known[i], true_id, templates, thresholds)

        if accept:
            TP += 1
        else:
            FN += 1


    FRR = FN / max(1, total)

    return FRR, TP, FN

def eval_unknown(E_impostor: torch.Tensor, y_impostor: torch.Tensor, grp_impostor: torch.Tensor, templates: dict, thresholds: dict) -> tuple[float, int, int, dict]:

    FP = 0
    TN = 0
    total = len(E_impostor)

    impostor_matches = defaultdict(list)

    for i in range(total):
        
        e = E_impostor[i]
        imp_key = f"user_{y_impostor[i]}_session_{grp_impostor[i]}"
        matched_users = [] 


        for claimed_id in templates.keys():
            accept, _ = verify_claim(e, claimed_id, templates, thresholds)
            
            if accept:
                matched_users.append(claimed_id)
                break
        if matched_users:

            FP += 1
            impostor_matches[imp_key] = matched_users
        else:

            TN += 1

    FAR = FP / max(1, total)
    
    return FAR, FP, TN, dict(impostor_matches)


    