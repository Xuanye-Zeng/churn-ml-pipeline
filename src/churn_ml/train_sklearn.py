from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_ml.evaluate import compute_metrics, summarize_cv_scores


def candidate_models(config: dict):
    models = {}
    logistic_config = config["models"]["logistic_regression"]
    if logistic_config.get("enabled", True):
        models["logistic_regression"] = LogisticRegression(
            max_iter=logistic_config.get("max_iter", 2000),
            random_state=config["random_state"],
        )

    forest_config = config["models"]["random_forest"]
    if forest_config.get("enabled", True):
        models["random_forest"] = RandomForestClassifier(
            n_estimators=forest_config.get("n_estimators", 200),
            max_depth=forest_config.get("max_depth", 8),
            min_samples_leaf=forest_config.get("min_samples_leaf", 2),
            random_state=config["random_state"],
        )

    return models


def train_sklearn_models(config: dict, preprocessor, X_train, X_test, y_train, y_test):
    results = {}
    cv_config = config.get("cross_validation", {})
    cv_enabled = cv_config.get("enabled", True)
    cv_folds = cv_config.get("folds", 5)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for model_name, estimator in candidate_models(config).items():
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)
        probabilities = model_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, predictions, probabilities)
        cv_summary = None

        if cv_enabled:
            # Clamp folds to minority class size to avoid StratifiedKFold errors
            class_counts = y_train.value_counts()
            max_supported_folds = int(class_counts.min())
            effective_folds = min(cv_folds, max_supported_folds)

            if effective_folds >= 2:
                cv_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", clone(preprocessor)),
                        ("model", estimator),
                    ]
                )
                cv = StratifiedKFold(
                    n_splits=effective_folds,
                    shuffle=True,
                    random_state=config["random_state"],
                )
                cv_results = cross_validate(
                    cv_pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=None,
                )
                cv_summary = summarize_cv_scores(
                    {
                        metric_name.replace("test_", ""): values
                        for metric_name, values in cv_results.items()
                        if metric_name.startswith("test_")
                    }
                )

        results[model_name] = {
            "framework": "scikit-learn",
            "metrics": metrics,
            "cross_validation": cv_summary,
            "artifact_type": "joblib",
            "model_object": model_pipeline,
            "test_probabilities": probabilities.tolist(),
        }

    return results
