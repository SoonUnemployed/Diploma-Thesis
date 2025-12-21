from collections import defaultdict
import torch 
import numpy as np

def aggregate_by_session(E: torch.Tensor, y: np.ndarray, sess: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    groups = defaultdict(list)

    for i in range(len(sess)):
        groups[(int(sess[i].split("_")[1]), int(sess[i].split("_")[-1]))].append(E[i])

    E_out, y_out, s_out = [], [], []
    for (subj, session), vecs in groups.items():
        
        v = torch.stack(vecs).mean(0)

        #v = torch.nn.functional.normalize(v, dim=0)

        E_out.append(v)
        y_out.append(subj)
        s_out.append(session)

    return (
        torch.stack(E_out),
        torch.tensor(y_out),
        torch.tensor(s_out)
    )

def aggregate_by_session_per_freq(E: torch.Tensor, y: np.ndarray, sess: np.ndarray, freq_array: np.ndarray, freq: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    groups = defaultdict(list)

    mask = freq_array == freq

    E = E[mask]
    y = y[mask]
    sess = sess[mask]

    for i in range(len(sess)):
        groups[(int(sess[i].split("_")[1]), int(sess[i].split("_")[-1]))].append(E[i])

    E_out, y_out, s_out = [], [], []
    for (subj, session), vecs in groups.items():
        
        v = torch.stack(vecs).mean(0)

        #v = torch.nn.functional.normalize(v, dim=0)

        E_out.append(v)
        y_out.append(subj)
        s_out.append(session)

    return (
        torch.stack(E_out),
        torch.tensor(y_out),
        torch.tensor(s_out)
    )