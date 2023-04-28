# Example function for pre-trained models

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import shapiro

import warnings
warnings.filterwarnings('ignore')

def roc_auc_conf_interval_bounds(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""

    labels_size = len(y)
    predictions = classifier.predict_proba(X)[:, 1]
    predictions_size = len(predictions)

    alpha = 1 - conf
    metric_values = []
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)
    
    for i in range(n_bootstraps):
        idx = rng.randint(0, labels_size, predictions_size)
        try:
            step_metric_score = roc_auc_score(y[idx], predictions[idx])
            metric_values.append(step_metric_score)
        except ValueError:
            pass
    
    p_value = shapiro(metric_values).pvalue

    if p_value < 0.05:
        lcb, ucb = np.quantile(metric_values, (alpha / 2)), np.quantile(metric_values, (1 - (alpha / 2)))
    else:
        std = np.std(metric_values)
        mean = np.average(metric_values)
        z = stats.norm.ppf(1 - alpha / 2)
        lcb, ucb = (mean - z * std), (mean + z * std)   

    return (lcb, ucb)