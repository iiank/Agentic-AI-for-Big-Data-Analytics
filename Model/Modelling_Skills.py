# Modelling_Skills.py
# Pure PySpark MLlib execution layer for the modelling agent.
# No LLM calls. No LangGraph. Only Spark + resampling logic.

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
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
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek


# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COL    = "Severity_Binary"
FEATURES_COL = "features"


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict] = {
 
    # ── Classification ────────────────────────────────────────────────────────
 
    "LogisticRegression": {
        "class":   LogisticRegression,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam":        [0.001, 0.01, 0.1, 1.0],
            "elasticNetParam": [0.0, 0.5, 1.0],
            "maxIter":         [100, 200],
        }
    },
 
    "RandomForestClassifier": {
        "class":   RandomForestClassifier,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "numTrees":            [100, 200, 300],
            "maxDepth":            [5, 10, 15],
            "minInstancesPerNode": [10, 50, 100],
        }
    },
 
    "GBTClassifier": {
        "class":   GBTClassifier,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "maxIter":  [20, 50],
            "maxDepth": [5, 10],
            "stepSize": [0.05, 0.1],
        }
    },
 
    "LinearSVC": {
        "class":   LinearSVC,
        "task":    "classification",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam": [0.001, 0.01, 0.1, 1.0],
            "maxIter":  [100, 200],
        }
    },
 
    # ── Regression ────────────────────────────────────────────────────────────
 
    "LinearRegression": {
        "class":   LinearRegression,
        "task":    "regression",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam":        [0.001, 0.01, 0.1, 1.0],
            "elasticNetParam": [0.0, 0.5, 1.0],
            "maxIter":         [100, 200],
        }
    },
 
    "RandomForestRegressor": {
        "class":   RandomForestRegressor,
        "task":    "regression",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "numTrees":            [100, 200, 300],
            "maxDepth":            [5, 10, 15],
            "minInstancesPerNode": [10, 50, 100],
        }
    },
 
    "GBTRegressor": {
        "class":   GBTRegressor,
        "task":    "regression",
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "maxIter":  [50, 100],
            "maxDepth": [5, 10],
            "stepSize": [0.05, 0.1],
        }
    },
 
}


# ── Training Function ─────────────────────────────────────────────────────────

def run_training(
    df: DataFrame,
    model_name: str,
    param_grid_spec: Dict[str, List],
    feature_cols: List[str],
    class_counts: Dict[int, int],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline for one model:
      1. Missing feature guard
      2. VectorAssembler
      3. 80/20 train/test split
      4. Downsample majority + SMOTETomek on train set
      5. 3-fold CrossValidator on balanced train
      6. Evaluate best model on held-out test set

    Args:
        df:              Raw engineered Spark DataFrame (not yet assembled).
        model_name:      Key from MODEL_REGISTRY.
        param_grid_spec: Dict of {param_name: [val1, val2, ...]} from LLM output.
        feature_cols:    Feature column names expected in df.
        seed:            Random seed for reproducibility.

    Returns:
        Dict with keys: model_name, auc, best_params, feature_importances, missing_features.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in MODEL_REGISTRY.")

    # ── 0. Create Severity_Binary if not present ──────────────────────────────
    if LABEL_COL not in df.columns:
        df = df.withColumn(LABEL_COL, when(col("Severity") >= 3, 1).otherwise(0))

    # ── 1. Missing feature guard ───────────────────────────────────────────────
    actual_cols    = set(df.columns)
    valid_features = [c for c in feature_cols if c in actual_cols]
    missing        = [c for c in feature_cols if c not in actual_cols]
    if missing:
        print(f"  [warn] Missing features skipped: {missing}")

    # ── 2. Assemble feature vector ─────────────────────────────────────────────
    if FEATURES_COL in df.columns:
        df = df.drop(FEATURES_COL)

    assembler    = VectorAssembler(inputCols=valid_features, outputCol=FEATURES_COL, handleInvalid="skip")
    df_assembled = assembler.transform(df).select(FEATURES_COL, LABEL_COL)

    # ── 3. Train / test split ──────────────────────────────────────────────────
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=seed)
    print(f"  Train: {train_df.count():,} rows | Test: {test_df.count():,} rows")

    # ── 4. Downsample + SMOTETomek ─────────────────────────────────────────────
    # print("  Downsampling majority class before resampling...")
    # fractions     = {0: 0.05, 1: 0.20}
    # sampled_train = train_df.stat.sampleBy(LABEL_COL, fractions, seed=seed)

    # print("  Converting to pandas for SMOTETomek...")
    # pdf        = sampled_train.toPandas()
    # X          = np.stack(pdf[FEATURES_COL].apply(lambda v: v.toArray()))
    # y          = pdf[LABEL_COL].values

    # print("  Applying SMOTETomek (may take a moment)...")
    # smt          = SMOTETomek(smote=SMOTE(random_state=seed), tomek=TomekLinks())
    # X_res, y_res = smt.fit_resample(X, y)

    # print("  Converting balanced data back to Spark...")
    # spark          = df.sparkSession
    # rdd            = spark.sparkContext.parallelize(zip(X_res, y_res)).map(
    #                    lambda x: (Vectors.dense(x[0]), int(x[1]))
    #                  )
    # schema         = StructType([
    #                    StructField(FEATURES_COL, VectorUDT(),   True),
    #                    StructField(LABEL_COL,    IntegerType(), True),
    #                  ])
    # balanced_train = spark.createDataFrame(rdd, schema)

    print("  Balancing classes...")
    minority_n   = int(min(class_counts.values()))
    majority_n   = int(max(class_counts.values()))
    majority_lbl = max(class_counts, key=class_counts.get)
    minority_lbl = min(class_counts, key=class_counts.get)
    fractions    = {
        majority_lbl: minority_n / majority_n,
        minority_lbl: 1.0,
    }
    print(f"  Class counts — majority: ~{majority_n:,} | minority: ~{minority_n:,}")
    print(f"  Downsampling majority to ~{minority_n:,} — balanced train: ~{minority_n * 2:,} rows")
 
    balanced_train = train_df.stat.sampleBy(LABEL_COL, fractions, seed=seed)

    # ── 5. Build model + CrossValidator ───────────────────────────────────────
    registry  = MODEL_REGISTRY[model_name]
    estimator = registry["class"](**registry["fixed"])

    grid_builder = ParamGridBuilder()
    for param_name, values in param_grid_spec.items():
        if hasattr(estimator, param_name):
            grid_builder = grid_builder.addGrid(getattr(estimator, param_name), values)
        else:
            print(f"  [warn] '{model_name}' has no param '{param_name}'. Skipping.")
    param_grid = grid_builder.build()
    print(f"  Grid combinations: {len(param_grid)} | CV folds: 3")

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=seed,
        parallelism=os.cpu_count(),
    )

    print(f"  Fitting CrossValidator for {model_name}...")
    cv_model   = cv.fit(balanced_train)
    best_model = cv_model.bestModel

    # ── 6. Evaluate on held-out test set ──────────────────────────────────────
    preds = cv_model.transform(test_df)
    auc   = round(evaluator.evaluate(preds), 4)
    print(f"  {model_name} Test AUC: {auc}")

    results = {
        "model_name":          model_name,
        "auc":                 auc,
        "best_params":         _extract_best_params(best_model, param_grid_spec),
        "feature_importances": _extract_feature_importances(best_model, valid_features),
        "missing_features":    missing,
    }

    # ── 7. Save to disk immediately ───────────────────────────────────────────
    try:
        _base = Path(__file__).parent
    except NameError:
        _base = Path.cwd()
    save_dir = _base / "saved_models" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model.write().overwrite().save(str(save_dir / "model"))
    with open(save_dir / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "feature_importances"}, f, indent=2)
    if results["feature_importances"]:
        with open(save_dir / "feature_importances.json", "w") as f:
            json.dump(results["feature_importances"], f, indent=2)
    print(f"  Saved to {save_dir}")
 
    return results


# ── Private Helpers ───────────────────────────────────────────────────────────

def _extract_best_params(best_model, param_grid_spec: Dict) -> Dict:
    best_params = {}
    for param_name in param_grid_spec.keys():
        getter = f"get{param_name[0].upper()}{param_name[1:]}"
        if hasattr(best_model, getter):
            val = getattr(best_model, getter)
            best_params[param_name] = val() if callable(val) else val
    return best_params


def _extract_feature_importances(best_model, feature_cols: List[str]) -> Optional[Dict]:
    if not hasattr(best_model, "featureImportances"):
        return None
    importances = best_model.featureImportances.toArray().tolist()
    paired = dict(zip(feature_cols, importances))
    return dict(sorted(paired.items(), key=lambda x: x[1], reverse=True))
