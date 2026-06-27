import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def plot_roc_curve(y_true, y_score, ax=None):
    """Plot a ROC curve from true labels and probability scores."""
    if ax is None:
        _, ax = plt.subplots()

    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC curve")
    return ax


def plot_precision_recall_curve(y_true, y_score, ax=None):
    """Plot a precision-recall curve from true labels and probability scores."""
    if ax is None:
        _, ax = plt.subplots()

    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Precision-recall curve")
    return ax
