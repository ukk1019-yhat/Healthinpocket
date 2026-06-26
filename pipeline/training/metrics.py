import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    classification_report,
)
from typing import Dict, List, Optional


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: Optional[np.ndarray] = None) -> Dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None and y_prob.shape[1] > 1:
        try:
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr")
                )
        except Exception:
            pass

    return metrics


def print_metrics(metrics: Dict, prefix: str = ""):
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score:  {metrics['f1_score']:.4f}")
    print(f"{prefix}Kappa:     {metrics['kappa']:.4f}")
    if "roc_auc" in metrics:
        print(f"{prefix}ROC AUC:   {metrics['roc_auc']:.4f}")
