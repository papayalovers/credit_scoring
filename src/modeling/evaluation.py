from typing import Union, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    recall_score, precision_score, accuracy_score, f1_score,
    confusion_matrix
)

# Create function to create auto selection threshold
def auto_selector_treshold(
        model: object,
        X_valid: pd.DataFrame,
        y_valid: Union[pd.Series, np.ndarray],
        strategy: str = 'recall',
        min_precision: float = None,
        thresholds: np.ndarray = np.arange(0.01, 1.0, 0.01)
) -> Tuple[float, pd.DataFrame]:
    
    # basic validation
    if len(X_valid) != len(y_valid):
        raise ValueError(f'X_valid and y_valid must have the same length')
    if strategy.lower() != 'recall':
        raise ValueError(f'Strategy only accepted for recall only')

    # Predict proba 
    y_proba = model.predict_proba(X_valid)[:, 1]

    results = list()
    
    for thres in thresholds:
        y_pred = (y_proba >= thres).astype(int)

        recall = recall_score(y_valid, y_pred, zero_division=0)
        precision = precision_score(y_valid, y_pred, zero_division=0)
        f1 = f1_score(y_valid, y_pred, zero_division=0)

        results.append({
            'threshold': thres,
            'recall': recall,
            'precision': precision,
            'f1-score': f1
        })
        
    df = pd.DataFrame(results)

    # apply precision constraint
    if min_precision is not None:
        if not isinstance(min_precision, (int, float)):
            raise ValueError("min_precision must be numeric")

        df = df[df['precision'] >= min_precision]

        if df.empty:
            raise ValueError("No threshold satisfies the precision constraint")

    # find best recall score with min precision values
    best_threshold = df.sort_values('recall',ascending=False).iloc[0]['threshold']

    return best_threshold, df

# fynction to eval on test setS
def evaluation_on_test_set(
        best_model: object,
        best_threshold: float,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray]
) -> dict:
    
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > best_threshold).astype(int)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=best_model.classes_)
    # extract value dari cm
    tn, fp, fn, tp = cm.ravel()
    # metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    # 
    results = {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return results
