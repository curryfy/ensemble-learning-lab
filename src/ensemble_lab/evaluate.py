import pandas as pd

from ensemble_lab.metrics import compute_binary_metrics


def evaluate_models(models, X_train, X_test, y_train, y_test, threshold=0.5):
    """Fit models and return a metrics table sorted by AUC."""
    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_score, threshold=threshold)

        rows.append({"model": model_name, **metrics})

    metrics_table = pd.DataFrame(rows)
    metrics_table = metrics_table[
        ["model", "auc", "ks", "pr_auc", "f1", "accuracy"]
    ]
    metrics_table = metrics_table.sort_values(
        by="auc",
        ascending=False,
    ).reset_index(drop=True)

    return metrics_table
