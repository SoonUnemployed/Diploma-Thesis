from sklearn.model_selection import StratifiedGroupKFold

def cross_validation() -> StratifiedGroupKFold:
    
    cv = StratifiedGroupKFold(
        n_splits = 5,
        shuffle = True,
        random_state = 19
    )

    return cv