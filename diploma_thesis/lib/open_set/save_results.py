import numpy as np
from pathlib import Path

def save_results(path: Path, TP: int, FN: int, FP: int, TN: int, FAR: float, FRR: float, thresholds: dict, imp_matches: dict, thrs: float):

    np.savez(

    path,
    acc_rate = thrs,
    TP = np.array(TP),
    FN = np.array(FN),
    FP = np.array(FP),
    TN = np.array(TN),
    FAR = np.array(FAR),
    FRR = np.array(FRR),

    thresholds = np.array(thresholds, dtype = object),
    impostor_matches = np.array(imp_matches, dtype = object),

)