# Modelling_Skills.py
# Pure PySpark MLlib execution layer for the modelling agent.
# No LLM calls. No LangGraph. Only Spark + resampling logic.

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, when, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import (
    LogisticRegression,
    LogisticRegressionModel,
    RandomForestClassifier,
    RandomForestClassificationModel,
    GBTClassifier,
    GBTClassificationModel,
    LinearSVC,
    LinearSVCModel,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COL    = "Severity_Binary"
FEATURES_COL = "features"


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict] = {
 
    # ── Classification ────────────────────────────────────────────────────────
 
    "LogisticRegression": {
        "class":               LogisticRegression,
        "load_class":          LogisticRegressionModel,
        "task":                "classification",
        "supports_probability": True,
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam":        [0.001, 0.01, 0.1, 1.0],
            "elasticNetParam": [0.0, 0.5, 1.0],
            "maxIter":         [100, 200],
        }
    },

    "RandomForestClassifier": {
        "class":               RandomForestClassifier,
        "load_class":          RandomForestClassificationModel,
        "task":                "classification",
        "supports_probability": True,
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL, "maxBins": 128},
        "tunable": {
            "numTrees":            [100, 200, 300],
            "maxDepth":            [5, 10, 15],
            "minInstancesPerNode": [10, 50, 100],
        }
    },

    "GBTClassifier": {
        "class":               GBTClassifier,
        "load_class":          GBTClassificationModel,
        "task":                "classification",
        "supports_probability": True,
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL, "featureSubsetStrategy": "sqrt", "maxBins": 128},
        "tunable": {
            "maxIter":             [10],
            "maxDepth":            [2]
        }
    },

    "LinearSVC": {
        "class":               LinearSVC,
        "load_class":          LinearSVCModel,
        "task":                "classification",
        "supports_probability": False,
        "fixed":   {"labelCol": LABEL_COL, "featuresCol": FEATURES_COL},
        "tunable": {
            "regParam": [0.001, 0.01, 0.1, 1.0],
            "maxIter":  [100, 200],
        }
    },
 
}


# ── Training Function ─────────────────────────────────────────────────────────

def run_training(
    train_df: DataFrame,
    test_df: DataFrame,
    model_name: str,
    param_grid_spec: Dict[str, List],
    feature_cols: List[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline for one model:
      1. Missing feature guard
      2. VectorAssembler
      3. Downsample majority + upsample minority on train set
      4. 3-fold CrossValidator on balanced train
      5. Evaluate best model on held-out test set

    Args:
        train_df:        Raw engineered training Spark DataFrame (not yet assembled).
        test_df:         Raw engineered testing Spark DataFrame (not yet assembled).
        model_name:      Key from MODEL_REGISTRY.
        param_grid_spec: Dict of {param_name: [val1, val2, ...]} from LLM output.
        feature_cols:    Feature column names expected in df.
        seed:            Random seed for reproducibility.

    Returns:
        Dict with keys: model_name, auc_roc, auc_pr, recall_class1, precision_class1,
        confusion_matrix, best_params, feature_importances, missing_features.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in MODEL_REGISTRY.")

    # ── 0. Create Severity_Binary if not present ──────────────────────────────
    if LABEL_COL not in train_df.columns:
        train_df = train_df.withColumn(LABEL_COL, when(col("Severity") >= 3, 1).otherwise(0))
    if LABEL_COL not in test_df.columns:
        test_df = test_df.withColumn(LABEL_COL, when(col("Severity") >= 3, 1).otherwise(0))

    # ── 1. Missing feature guard ───────────────────────────────────────────────
    actual_cols    = set(train_df.columns)
    valid_features = [c for c in feature_cols if c in actual_cols]
    missing        = [c for c in feature_cols if c not in actual_cols]
    if missing:
        print(f"  [warn] Missing features skipped: {missing}")

    # ── 2. Assemble feature vector ─────────────────────────────────────────────
    if FEATURES_COL in train_df.columns:
        train_df = train_df.drop(FEATURES_COL)
    if FEATURES_COL in test_df.columns:
        test_df = test_df.drop(FEATURES_COL)
    assembler = VectorAssembler(
        inputCols=valid_features,
        outputCol=FEATURES_COL,
        handleInvalid="skip"
    )

    train_df_assembled = assembler.transform(train_df).select(FEATURES_COL, LABEL_COL)
    test_df_assembled  = assembler.transform(test_df).select(FEATURES_COL, LABEL_COL)

    # Strip ML attribute metadata by recreating the vector
    to_dense_udf = udf(lambda v: Vectors.dense(v.toArray().tolist()), VectorUDT())

    train_df_assembled = train_df_assembled.withColumn(
        FEATURES_COL, to_dense_udf(col(FEATURES_COL))
    )
    test_df_assembled = test_df_assembled.withColumn(
        FEATURES_COL, to_dense_udf(col(FEATURES_COL))
    )

    # ── 3. Balance classes ────────────────────────────────────────────────────
    print("  Balancing classes...")
    train_class_counts = train_df_assembled.groupBy(LABEL_COL).count().collect()
    train_counts = {int(row[LABEL_COL]): row["count"] for row in train_class_counts}
    minority_lbl = min(train_counts, key=train_counts.get)
    majority_lbl = max(train_counts, key=train_counts.get)
    minority_n   = train_counts[minority_lbl]
    majority_n   = train_counts[majority_lbl]
    
    print(f"  Train counts - majority: {majority_n:,} | minority: {minority_n:,}")

    # Downsample majority to match minority exactly (50:50)
    fraction_majority = minority_n / majority_n
    balanced_train = train_df_assembled.stat.sampleBy(
        LABEL_COL,
        fractions={majority_lbl: fraction_majority, minority_lbl: 1.0},
        seed=seed
    )
    balanced_train = balanced_train.limit(10)

    print(f"  Downsampling majority to ~{minority_n:,} — balanced train: ~{minority_n * 2:,} rows")

    # ── 4. Build model + CrossValidator ───────────────────────────────────────
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

    cv_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=cv_evaluator,
        numFolds=3,
        seed=seed,
        parallelism=os.cpu_count(),
    )

    print(f"  Fitting CrossValidator for {model_name}...")
    cv_model   = cv.fit(balanced_train)
    best_model = cv_model.bestModel

    # ── 6. Evaluate on held-out test set ──────────────────────────────────────
    preds = cv_model.transform(test_df_assembled)

    # AUC-ROC and AUC-PR
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    auc_roc = round(evaluator_roc.evaluate(preds), 4)
    auc_pr  = round(evaluator_pr.evaluate(preds),  4)

    # Recall and precision for class 1 (at default 0.5 threshold)
    recall_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction",
        metricName="recallByLabel", metricLabel=1.0,
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction",
        metricName="precisionByLabel", metricLabel=1.0,
    )
    recall_class1    = round(recall_eval.evaluate(preds),    4)
    precision_class1 = round(precision_eval.evaluate(preds), 4)

    # Confusion matrix at default 0.5 threshold
    tp = preds.filter((F.col("prediction") == 1.0) & (F.col(LABEL_COL) == 1)).count()
    fn = preds.filter((F.col("prediction") == 0.0) & (F.col(LABEL_COL) == 1)).count()
    fp = preds.filter((F.col("prediction") == 1.0) & (F.col(LABEL_COL) == 0)).count()
    tn = preds.filter((F.col("prediction") == 0.0) & (F.col(LABEL_COL) == 0)).count()

    print(f"  {model_name} | AUC-ROC: {auc_roc} | AUC-PR: {auc_pr} | Recall@1: {recall_class1} | Precision@1: {precision_class1}")
    print(f"  Confusion matrix | TP: {tp:,}  FN: {fn:,}  FP: {fp:,}  TN: {tn:,}")

    results = {
        "model_name":          model_name,
        "auc_roc":             auc_roc,
        "auc_pr":              auc_pr,
        "recall_class1":       recall_class1,
        "precision_class1":    precision_class1,
        "confusion_matrix": {
            "true_positives":  tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives":  tn,
        },
        "best_params":         _extract_best_params(best_model, param_grid_spec),
        "feature_importances": _extract_feature_importances(best_model, valid_features),
        "missing_features":    missing,
    }

    # ── 6. Save to disk immediately ───────────────────────────────────────────
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


# ── Threshold Tuning ─────────────────────────────────────────────────────────

def run_threshold_tuning(
    test_df: DataFrame,
    model_name: str,
    feature_cols: List[str],
    thresholds: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sweeps decision thresholds on the held-out test set for the saved winner model.
    Uses a single Spark aggregation job across all thresholds (no per-threshold scans).

    Args:
        df:           Raw engineered Spark DataFrame (same as passed to run_training).
        model_name:   Key from MODEL_REGISTRY — loads from saved_models/{model_name}/model.
        feature_cols: Same feature columns used during training.
        thresholds:   List of thresholds to try. Defaults to [0.20 ... 0.50].
        seed:         Must match the seed used in run_training to reproduce the same test split.

    Returns:
        Dict with best_threshold, metrics at each threshold, and full breakdown.
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    if not MODEL_REGISTRY[model_name].get("supports_probability", False):
        print(f"  [warn] {model_name} does not support probability output — threshold tuning skipped.")
        return {"skipped": True, "reason": f"{model_name} does not output a probability column"}

    # ── 1. Recreate same test set ──────────────────────────────────────────────
    if LABEL_COL not in test_df.columns:
        test_df = test_df.withColumn(LABEL_COL, F.when(F.col("Severity") >= 3, 1).otherwise(0))

    actual_cols    = set(test_df.columns)
    valid_features = [c for c in feature_cols if c in actual_cols]

    if FEATURES_COL in test_df.columns:
        test_df = test_df.drop(FEATURES_COL)

    assembler    = VectorAssembler(inputCols=valid_features, outputCol=FEATURES_COL, handleInvalid="skip")
    test_df = assembler.transform(test_df).select(FEATURES_COL, LABEL_COL)

    # ── 2. Load saved model ────────────────────────────────────────────────────
    try:
        _base = Path(__file__).parent
    except NameError:
        _base = Path.cwd()

    save_path  = _base / "saved_models" / model_name / "model"
    load_class = MODEL_REGISTRY[model_name]["load_class"]
    model      = load_class.load(str(save_path))
    print(f"  Loaded {model_name} from {save_path}")

    # ── 3. Get predictions with probability ───────────────────────────────────
    preds = model.transform(test_df)

    if "probability" not in preds.columns:
        print(f"  [warn] No probability column found — threshold tuning skipped.")
        return {"skipped": True, "reason": "probability column not present in predictions"}

    # ── 4. Single aggregation across all thresholds ───────────────────────────
    preds = preds.withColumn("prob1", F.col("probability")[1])

    for t in thresholds:
        key = int(t * 100)
        preds = preds.withColumn(f"pred_{key}", (F.col("prob1") >= t).cast("int"))

    agg_exprs = []
    for t in thresholds:
        key = int(t * 100)
        agg_exprs += [
            F.sum((F.col(f"pred_{key}") == 1) & (F.col(LABEL_COL) == 1)).alias(f"tp_{key}"),
            F.sum((F.col(f"pred_{key}") == 0) & (F.col(LABEL_COL) == 1)).alias(f"fn_{key}"),
            F.sum((F.col(f"pred_{key}") == 1) & (F.col(LABEL_COL) == 0)).alias(f"fp_{key}"),
            F.sum((F.col(f"pred_{key}") == 0) & (F.col(LABEL_COL) == 0)).alias(f"tn_{key}"),
        ]

    counts = preds.agg(*agg_exprs).collect()[0]

    # ── 5. Build per-threshold results ────────────────────────────────────────
    results_by_threshold = []
    for t in thresholds:
        key = int(t * 100)
        tp  = counts[f"tp_{key}"] or 0
        fn  = counts[f"fn_{key}"] or 0
        fp  = counts[f"fp_{key}"] or 0
        tn  = counts[f"tn_{key}"] or 0

        recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0

        results_by_threshold.append({
            "threshold":       t,
            "recall_class1":   recall,
            "precision_class1": precision,
            "confusion_matrix": {
                "true_positives":  int(tp),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives":  int(tn),
            },
        })
        print(f"  threshold={t} | recall={recall} | precision={precision} | TP={tp:,} FN={fn:,}")

    # ── 6. Pick best threshold: highest recall with non-zero precision ─────────
    valid   = [r for r in results_by_threshold if r["precision_class1"] > 0]
    best    = max(valid, key=lambda r: r["recall_class1"]) if valid else results_by_threshold[-1]
    print(f"  Best threshold: {best['threshold']} (recall={best['recall_class1']}, precision={best['precision_class1']})")

    return {
        "model_name":         model_name,
        "best_threshold":     best["threshold"],
        "best_metrics":       best,
        "all_thresholds":     results_by_threshold,
        "skipped":            False,
    }


# ── Full Evaluation ───────────────────────────────────────────────────────────

def run_full_evaluation(
    df: DataFrame,
    model_name: str,
    feature_cols: List[str],
    seed: int = 42,
) -> None:
    """
    Generates full evaluation artefacts for a saved model on the held-out test set:
      - Classification report (precision / recall / F1 for both classes)
      - Confusion matrix heatmap
      - ROC curve (with AUC-ROC)
      - Precision-Recall curve (with AUC-PR)

    Plots are saved to saved_models/{model_name}/.
    Requires matplotlib and sklearn (post-hoc visualisation only, not part of training).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )

    print(f"\n[Full Evaluation] {model_name}")

    # ── 1. Recreate same test set ──────────────────────────────────────────────
    if LABEL_COL not in df.columns:
        df = df.withColumn(LABEL_COL, F.when(F.col("Severity") >= 3, 1).otherwise(0))

    actual_cols    = set(df.columns)
    valid_features = [c for c in feature_cols if c in actual_cols]

    if FEATURES_COL in df.columns:
        df = df.drop(FEATURES_COL)

    assembler    = VectorAssembler(inputCols=valid_features, outputCol=FEATURES_COL, handleInvalid="skip")
    df_assembled = assembler.transform(df).select(FEATURES_COL, LABEL_COL)
    _, test_df   = df_assembled.randomSplit([0.8, 0.2], seed=seed)

    # ── 2. Load saved model ────────────────────────────────────────────────────
    try:
        _base = Path(__file__).parent
    except NameError:
        _base = Path.cwd()

    save_dir   = _base / "saved_models" / model_name
    load_class = MODEL_REGISTRY[model_name]["load_class"]
    model      = load_class.load(str(save_dir / "model"))

    # ── 3. Get predictions ─────────────────────────────────────────────────────
    prob1_udf = udf(lambda v: float(v[1]), DoubleType())
    preds = model.transform(test_df)
    preds = preds.withColumn("prob1", prob1_udf(F.col("probability")))

    has_probability = "probability" in preds.columns
    if has_probability:

        preds_pd = preds.select(
            F.col("prediction").cast("int").alias("pred"),
            F.col(LABEL_COL).alias("label"),
            F.col("probability")[1].alias("prob1"),
        ).toPandas()
    else:
        preds_pd = preds.select(
            F.col("prediction").cast("int").alias("pred"),
            F.col(LABEL_COL).alias("label"),
        ).toPandas()

    y_true = preds_pd["label"].values
    y_pred = preds_pd["pred"].values

    # ── 4. Classification report ───────────────────────────────────────────────
    print(f"\n  Classification Report — {model_name}")
    print("  " + "-" * 50)
    report = classification_report(
        y_true, y_pred,
        target_names=["Low Severity (0)", "High Severity (1)"],
    )
    for line in report.splitlines():
        print(f"  {line}")

    # ── 5. Confusion matrix heatmap ────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Pred: Low", "Pred: High"],
        yticklabels=["Act: Low", "Act: High"],
        title=f"Confusion Matrix — {model_name}",
        ylabel="Actual", xlabel="Predicted",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Saved: {cm_path}")

    if not has_probability:
        print(f"  [info] {model_name} has no probability output — ROC and PR curves skipped.")
        return

    y_prob = preds_pd["prob1"].values

    # ── 6. ROC curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC-ROC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(
        xlabel="False Positive Rate", ylabel="True Positive Rate",
        title=f"ROC Curve — {model_name}",
        xlim=[0, 1], ylim=[0, 1.02],
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = save_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  Saved: {roc_path}")

    # ── 7. Precision-Recall curve ─────────────────────────────────────────────
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, lw=2, label=f"AUC-PR = {auc_pr:.4f}")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1, label=f"Baseline = {baseline:.3f}")
    ax.set(
        xlabel="Recall", ylabel="Precision",
        title=f"Precision-Recall Curve — {model_name}",
        xlim=[0, 1], ylim=[0, 1.02],
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    pr_path = save_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"  Saved: {pr_path}")


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
