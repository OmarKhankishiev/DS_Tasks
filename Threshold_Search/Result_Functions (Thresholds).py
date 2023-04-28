# Result functions for pre-trained models

from typing import Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Precision-Recall Curve)"""
    
    def custom_pr_curve(y_true, y_prob):
        
        precision = np.zeros_like(y_true, dtype=np.float64)
        recall = np.zeros_like(y_true, dtype = np.float64)
    
        tp = 0
        fp = 0
        fn = np.sum(y_true == 1)

        for i in range(len(y_true)):
            if y_true[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
            precision[i] = tp / (tp + fp)
            recall[i] = tp / (tp + fn)
        return precision, recall, y_prob

    precision, recall, proba = custom_pr_curve(y_true, y_prob)
    metrics_values_df = pd.DataFrame({
        'proba': proba,
        'precision': precision,
        'recall': recall
    })

    threshold_proba, max_recall = metrics_values_df[metrics_values_df['precision'] >= min_precision].sort_values(['recall', 'proba'], ascending = [False, False])[['proba', 'recall']].head(1).values[0]  

    return threshold_proba, max_recall


def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    def custom_sr_curve(y_true, y_prob):

        desc_score_idx = np.argsort(y_prob)[::-1]
        y_prob = y_prob[desc_score_idx]
        y_true = y_true[desc_score_idx]
    
        recall = np.zeros_like(y_true, dtype = np.float64)
        specificity = np.zeros_like(y_true, dtype = np.float64)

        tp = 0
        fp = 0
        fn = np.sum(y_true == 1)
        tn = np.sum(y_true == 0)

        for i in range(len(y_true)):
            if y_true[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
            recall[i] = tp / (tp + fn)
            specificity[i] = tn / (tn + fp)

        return specificity, recall, y_prob

    specificity, recall, proba = custom_sr_curve(y_true, y_prob)
    metrics_values_df = pd.DataFrame({
        'proba': proba,
        'specificity': specificity,
       'recall': recall
    })

    threshold_proba, max_recall = metrics_values_df[metrics_values_df['specificity'] >= min_specificity].sort_values(['recall', 'proba'], ascending = [False, False])[['proba', 'recall']].head(1).values[0]

    return threshold_proba, max_recall