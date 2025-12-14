from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(preds, labels, average = 'macro'):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds)
    }