from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true, y_score, threshold=0.5):
    """Compute common binary classification metrics from probability scores."""
    y_pred = y_score >= threshold
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = max(tpr - fpr)

    return {
        "auc": roc_auc_score(y_true, y_score),
        "ks": ks,
        "pr_auc": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
