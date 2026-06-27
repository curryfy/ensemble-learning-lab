from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_stage1_models(random_state=42):
    """Return simple baseline classifiers for Stage 1 experiments."""
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=4,
        random_state=random_state,
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=random_state),
        n_estimators=50,
        random_state=random_state,
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
    )

    gradient_boosting = GradientBoostingClassifier(random_state=random_state)

    hist_gradient_boosting = HistGradientBoostingClassifier(
        random_state=random_state,
    )

    stacking = StackingClassifier(
        estimators=[
            ("logistic_regression", logistic_regression),
            ("random_forest", random_forest),
            ("gradient_boosting", gradient_boosting),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
    )

    return {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "Bagging": bagging,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "Hist Gradient Boosting": hist_gradient_boosting,
        "Stacking": stacking,
    }
