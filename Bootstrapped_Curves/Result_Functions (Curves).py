# Result functions for pre-trained models

from typing import Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)"""
    alpha = 1 - conf

    rng_seed = 42
    rng = np.random.RandomState(rng_seed)

    def custom_precision_recall_curve(y_true, y_prob):
        
        desc_score_idx = np.argsort(y_prob)[::-1]
        y_prob = y_prob[desc_score_idx]
        y_true = y_true[desc_score_idx]

        precision = np.zeros(len(y_true))
        recall = np.zeros(len(y_true))
    
        tp = 0
        fp = 0
        fn = np.sum(y_true == 1)

        for i in range(len(y_true)):
            if y_true[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
            recall[i] = tp / (tp + fn)
            precision[i] = tp / (tp + fp)

        return np.array(precision), np.array(recall)

    precisions = []
    recalls = []

    for i in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_prob))
        y_true_b, y_prob_b = y_true[idx], y_prob[idx]
        step_precision, step_recall = custom_precision_recall_curve(y_true_b, y_prob_b)
        precisions.append(step_precision)
        recalls.append(step_recall)

    precisions = np.array(precisions)
    precisions = precisions[~np.isnan(precisions).any(axis = 1)]
    precision = np.mean(precisions, axis = 0)

    recalls = np.array(recalls)
    recalls = recalls[~np.isnan(recalls).any(axis = 1)]
    recall = np.mean(recalls, axis = 0)

    precision_lcb = np.quantile(precisions, (alpha / 2), axis = 0)
    precision_ucb = np.quantile(precisions, (1 - (alpha / 2)), axis = 0)

    recall_lcb = np.quantile(recalls, (alpha / 2), axis = 0)
    recall_ucb = np.quantile(recalls, (1 - (alpha / 2)), axis = 0)

    recall, precision = np.clip(recall, 0, 1), np.clip(precision, 0, 1)
    precision_lcb, precision_ucb = np.clip(precision_lcb, 0, 1), np.clip(precision_ucb, 0, 1)
    recall_lcb, recall_ucb = np.clip(recall_lcb, 0, 1), np.clip(recall_ucb, 0, 1)
    
    return recall, recall_lcb, recall_ucb, precision, precision_lcb, precision_ucb

def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)"""

    alpha = 1 - conf
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)

    def custom_specificity_recall_curve(y_true, y_prob):
        
        desc_score_idx = np.argsort(y_prob)[::-1]
        y_prob = y_prob[desc_score_idx]
        y_true = y_true[desc_score_idx]
    
        specificity = np.zeros(len(y_true))
        recall = np.zeros(len(y_true))
    
        fp = 0
        tp = 0
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
        
        return np.array(specificity), np.array(recall)

    specificities = []
    recalls = []

    for i in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_prob))
        y_true_b, y_prob_b = y_true[idx], y_prob[idx]
        step_specificity, step_recall = custom_specificity_recall_curve(y_true_b, y_prob_b)
        specificities.append(step_specificity)
        recalls.append(step_recall)

    specificities = np.array(specificities)
    specificities = specificities[~np.isnan(specificities).any(axis = 1)]
    specificity = np.mean(specificities, axis = 0)

    recalls = np.array(recalls)
    recalls = recalls[~np.isnan(recalls).any(axis = 1)]
    recall = np.mean(recalls, axis = 0)

    specificity_lcb = np.quantile(specificities, (alpha / 2), axis = 0)
    specificity_ucb = np.quantile(specificities, (1 - (alpha / 2)), axis = 0)

    recall_lcb = np.quantile(recalls, (alpha / 2), axis = 0)
    recall_ucb = np.quantile(recalls, (1 - (alpha / 2)), axis = 0)

    recall, specificity = np.clip(recall, 0, 1), np.clip(specificity, 0, 1)
    specificity_lcb, specificity_ucb = np.clip(specificity_lcb, 0, 1), np.clip(specificity_ucb, 0, 1)
    recall_lcb, recall_ucb = np.clip(recall_lcb, 0, 1), np.clip(recall_ucb, 0, 1)
   
    return recall, recall_lcb, recall_ucb, specificity, specificity_lcb, specificity_ucb