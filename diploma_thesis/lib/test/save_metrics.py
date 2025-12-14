import numpy as np
from pathlib import Path

def save_metrics(results: dict, y_pred: np.array, y: np.array, model: str, output: Path):

    np.savez(
        
        output / f"{model}_closed_set_metrics.npz",
        accuracy = results["accuracy"],
        precision = results["precision"],
        recall = results["recall"],
        f1 = results["f1"],

        y_true = y,
        y_pred = y_pred,

        confusion_matrix = results["confusion_matrix"],

        model = model,
        
        number_of_users = len(set(y)),
        chance = 1.0 / len(set(y))

    )