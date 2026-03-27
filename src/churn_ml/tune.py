"""Hyperparameter tuning with Ray Tune.

Each trial evaluates a parameter combination using stratified
cross-validation on the training set. Results feed back into
the main training config so the final models use tuned params.
"""

from __future__ import annotations

from copy import deepcopy

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    import ray
    from ray import tune
    from ray.tune import RunConfig, TuneConfig, Tuner

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def _build_search_spaces():
    """Define per-model hyperparameter search spaces.

    loguniform for C: regularization strength varies over orders of magnitude.
    choice for tree params: discrete values that are common in practice.
    """
    return {
        "logistic_regression": {
            "C": tune.loguniform(1e-2, 1e1),
        },
        "random_forest": {
            "n_estimators": tune.choice([50, 100, 200, 300]),
            "max_depth": tune.choice([4, 6, 8, 10, None]),
            "min_samples_leaf": tune.choice([1, 2, 4, 8]),
        },
    }


def _build_model(model_name: str, params: dict, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=random_state,
            class_weight="balanced",
        )
    raise ValueError(f"Unknown model: {model_name}")


def _trainable(
    trial_config,
    model_name,
    preprocessor,
    X_train,
    y_train,
    random_state,
    cv_folds,
):
    """Single Ray Tune trial: build a model pipeline, run CV, report F1."""
    model = _build_model(model_name, trial_config, random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    return {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}


def run_tuning(config: dict, preprocessor, X_train, y_train) -> dict:
    """Run hyperparameter search for enabled sklearn models.

    Returns a dict mapping model names to tuning results,
    or empty dict if tuning is disabled or Ray is unavailable.
    """
    tune_cfg = config.get("tuning", {})
    if not tune_cfg.get("enabled", False):
        return {}
    if not RAY_AVAILABLE:
        print("Ray not installed, skipping tuning.")
        return {}

    num_samples = tune_cfg.get("num_samples", 20)
    cv_folds = tune_cfg.get("cv_folds", 3)
    random_state = config["random_state"]
    search_spaces = _build_search_spaces()

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    results = {}
    try:
        for model_name, search_space in search_spaces.items():
            model_cfg = config["models"].get(model_name, {})
            if not model_cfg.get("enabled", True):
                continue

            # tune.with_parameters puts large objects (data, preprocessor) in
            # the Ray object store so they're shared across trials efficiently.
            trainable_fn = tune.with_parameters(
                _trainable,
                model_name=model_name,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                random_state=random_state,
                cv_folds=cv_folds,
            )

            tuner = Tuner(
                trainable_fn,
                param_space=search_space,
                tune_config=TuneConfig(
                    num_samples=num_samples,
                    metric="f1_mean",
                    mode="max",
                ),
                run_config=RunConfig(verbose=0),
            )
            result_grid = tuner.fit()
            best = result_grid.get_best_result(metric="f1_mean", mode="max")

            # Sanitize params: convert numpy types to native Python for JSON
            best_params = {k: _to_native(v) for k, v in best.config.items()}

            results[model_name] = {
                "best_params": best_params,
                "best_f1_mean": round(best.metrics["f1_mean"], 4),
                "best_f1_std": round(best.metrics["f1_std"], 4),
                "num_trials": len(result_grid),
            }
            print(
                f"  {model_name}: best CV F1 = {results[model_name]['best_f1_mean']}"
                f" (std {results[model_name]['best_f1_std']})"
            )
    finally:
        ray.shutdown()

    return results


def _to_native(value):
    """Convert numpy/ray types to JSON-safe Python types."""
    if hasattr(value, "item"):
        return value.item()
    return value


def apply_tuned_params(config: dict, tune_results: dict) -> dict:
    """Merge tuned hyperparameters back into the pipeline config."""
    updated = deepcopy(config)
    for model_name, result in tune_results.items():
        if model_name in updated["models"]:
            updated["models"][model_name].update(result["best_params"])
    return updated
