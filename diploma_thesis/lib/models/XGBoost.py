from xgboost import XGBClassifier
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from lib.test.metrics import evaluate_model

def XGBoost_Classifier(y: np.array) -> XGBClassifier:
    
    model = XGBClassifier(
        n_estimators = 300,
        max_depth = 6,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        reg_alpha = 1.4,
        reg_lambda = 2.0,
        eval_metric = "mlogloss",
        objective = "multi:softprob",
        num_class = len(np.unique(y)),
        tree_method = "hist"
    )
    '''
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
    )
    '''

    return model

def train_xgboost(model: XGBClassifier, X: np.array, y: np.array, X_val: np.array, y_val: np.array) -> tuple[XGBClassifier, float]:
    
    model.fit(
        X, y,
        eval_set = [(X_val, y_val)],
        verbose = False
    )

    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    print("Training Accuracy:", train_acc)

    return model, train_acc


def test_xgboost(model: XGBClassifier, X: np.array, y: np.array, model_path: Path) -> tuple[dict, np.array]:
    
    m_path = model_path / ("xgboost.json")
    model.load_model(m_path)

    y_pred_closed = model.predict(X)
    acc_closed = accuracy_score(y, y_pred_closed)

    xgb_metrics = evaluate_model(y_pred_closed, y)

    print("Closed-set XGBoost accuracy:", acc_closed)
    print(classification_report(y, y_pred_closed))

    print("\n===== XGBoost Metrics =====")
    for k, v in xgb_metrics.items():
        print(f"{k}: \n{v}\n")

    return xgb_metrics, y_pred_closed