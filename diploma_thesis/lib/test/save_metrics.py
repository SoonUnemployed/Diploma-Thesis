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

def save_metrics_per_frequency(results: dict, y: np.array, model: str, output: Path):

    for f, res_f in results.items():

        metrics = res_f["metrics"]
        y_pred = res_f["predictions"]

        np.savez(
            
            output / f"{model}_closed_set_metrics_{f}.npz",

            frequency = f,
            accuracy = metrics["accuracy"],
            precision = metrics["precision"],
            recall = metrics["recall"],
            f1 = metrics["f1"],
            confusion_matrix = metrics["confusion_matrix"],

            y_true = y,
            y_pred = y_pred,

            model = model,

            number_of_users = len(set(y)),
            chance = 1.0 / len(set(y))

        )
    
    return