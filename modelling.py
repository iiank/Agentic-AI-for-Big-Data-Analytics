# modelling_pyspark_skills.py
# Pure PySpark MLlib execution layer for the modelling agent.
# No LLM calls. No LangGraph. Only Spark logic.
# The agent calls functions from this file based on LLM decisions.

from typing import Dict, List, Optional, Tuple, Any

from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    LinearSVC,
)
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COL    = "Severity_Binary"
FEATURES_COL = "features"


# ── Model Registry ────────────────────────────────────────────────────────────
# Defines every model the agent may select.
# Classification and regression variants are both registered so the LLM
# can explicitly reason about and exclude the inapplicable task type.
# The agent reads the registry keys when building the LLM context.

MODEL_REGISTRY: Dict[str, Dict] = {

    # ── Classification ────────────────────────────────────────────────────────
    "LogisticRegression": {
        "class":   LogisticRegression,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam":        [0.01, 0.1, 1.0],
            "elasticNetParam": [0.0, 0.5, 1.0],
            "maxIter":         [50, 100, 200],
        },
    },
    "RandomForestClassifier": {
        "class":   RandomForestClassifier,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "numTrees":            [50, 100, 200],
            "maxDepth":            [5, 10],
            "minInstancesPerNode": [10, 50, 100],
        },
    },
    "GBTClassifier": {
        "class":   GBTClassifier,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "maxIter":  [20, 50],
            "maxDepth": [5, 10],
            "stepSize": [0.05, 0.1],
        },
    },
    "LinearSVC": {
        "class":   LinearSVC,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam": [0.01, 0.1],
            "maxIter":  [50, 100, 200],
        },
    },

    # ── Regression (agent-aware; excluded for binary classification tasks) ────
    "LinearRegression": {
        "class":   LinearRegression,
        "task":    "regression",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam":        [0.01, 0.1, 1.0],
            "elasticNetParam": [0.0, 0.5, 1.0],
            "maxIter": [100, 200]
        },
    },
    "RandomForestRegressor": {
        "class":   RandomForestRegressor,
        "task":    "regression",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "numTrees": [50, 100],
            "maxDepth": [5, 10],
            "minInstancesPerNode": [10, 50] 
        },
    },
    "GBTRegressor": {
        "class":   GBTRegressor,
        "task":    "regression",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "maxIter":  [20, 50],
            "maxDepth": [5, 10],
            "stepSize": [0.05, 0.1],
        },
    },
}


# ── Training Function ─────────────────────────────────────────────────────────

def run_training(
    df: DataFrame,
    model_name: str,
    param_grid_spec: Dict[str, List],
    feature_cols: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Instantiate the model from the registry, build a param grid from the
    LLM-proposed values, run TrainValidationSplit, and return results.

    Uses TrainValidationSplit (single 80/20 holdout) instead of CrossValidator
    to keep runtime tractable on a local Spark session with 5M rows.

    Args:
        df:              Engineered Spark DataFrame with 'features' and LABEL_COL.
        model_name:      Key from MODEL_REGISTRY (chosen by the LLM).
        param_grid_spec: Dict of {param_name: [val1, val2, ...]} from LLM output.
        feature_cols:    List of feature names for importance extraction.
        train_ratio:     Fraction of data used for training (default 0.8).
        seed:            Random seed for reproducibility.

    Returns:
        Dict with keys: model_name, auc, best_params, feature_importances.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in MODEL_REGISTRY.")

    registry  = MODEL_REGISTRY[model_name]
    estimator = registry["class"](**registry["fixed"])

    # Build param grid from LLM-proposed values
    grid_builder = ParamGridBuilder()
    for param_name, values in param_grid_spec.items():
        if hasattr(estimator, param_name):
            grid_builder = grid_builder.addGrid(
                getattr(estimator, param_name), values
            )
        else:
            print(f"  [warn] '{model_name}' has no param '{param_name}'. Skipping.")
    param_grid = grid_builder.build()

    print(f"  Grid combinations to evaluate: {len(param_grid)}")

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    tvs = TrainValidationSplit(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=train_ratio,
        seed=seed,
        parallelism=2,
    )

    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    print(f"  Train: {train_df.count():,} rows | Test: {test_df.count():,} rows")
    print("  Fitting... (this may take several minutes on local Spark)")

    tvs_model  = tvs.fit(train_df)
    best_model = tvs_model.bestModel
    preds      = tvs_model.transform(test_df)
    auc        = round(evaluator.evaluate(preds), 4)

    best_params = _extract_best_params(best_model, param_grid_spec)
    feature_importances = _extract_feature_importances(best_model, feature_cols)

    return {
        "model_name":           model_name,
        "auc":                  auc,
        "best_params":          best_params,
        "feature_importances":  feature_importances,
    }


# ── Private Helpers ───────────────────────────────────────────────────────────

def _extract_best_params(best_model, param_grid_spec: Dict) -> Dict:
    """Pull the winning hyperparameter values off the fitted best model."""
    best_params = {}
    for param_name in param_grid_spec.keys():
        getter_name = f"get{param_name[0].upper()}{param_name[1:]}"
        if hasattr(best_model, getter_name):
            best_params[param_name] = getattr(best_model, getter_name)()
    return best_params


def _extract_feature_importances(best_model, feature_cols: List[str]) -> Optional[Dict]:
    """Return sorted feature importances for tree-based models. None otherwise."""
    if not hasattr(best_model, "featureImportances"):
        return None
    importances = best_model.featureImportances.toArray().tolist()
    paired = dict(zip(feature_cols, importances))
    return dict(sorted(paired.items(), key=lambda x: x[1], reverse=True))