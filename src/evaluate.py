

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
import sys
import os
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)
from src import config


def pr_auc(y_true, y_proba) -> float:
    """Average precision = area under the Precision-Recall curve."""
    return average_precision_score(y_true, y_proba)


def recall_at_fpr(y_true, y_proba, target_fpr: float = config.TARGET_FPR) -> float:
    """
    Finds the largest recall achievable while keeping the False Positive
    Rate at or below target_fpr.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # tpr IS recall for the positive class
    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def full_report(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    One-stop function: given true labels + predicted probabilities,
    returns every metric we care about for this project.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    report = {
        "threshold_used": threshold,
        "pr_auc": round(pr_auc(y_true, y_proba), 4),
        "recall_at_target_fpr": round(recall_at_fpr(y_true, y_proba), 4),
        "target_fpr": config.TARGET_FPR,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    return report


def print_report(report: dict, title: str = "Model Report"):
    print(f"\n===== {title} =====")
    print(f"PR-AUC:                    {report['pr_auc']}")
    print(f"Recall @ FPR<={report['target_fpr']}:      {report['recall_at_target_fpr']}")
    print(f"Threshold used:            {report['threshold_used']}")
    cm = report["confusion_matrix"]
    print(f"Confusion matrix -> TN:{cm['TN']}  FP:{cm['FP']}  FN:{cm['FN']}  TP:{cm['TP']}")
    cr = report["classification_report"]["1"]
    print(f"Fraud class -> precision:{cr['precision']:.3f}  recall:{cr['recall']:.3f}  f1:{cr['f1-score']:.3f}")
