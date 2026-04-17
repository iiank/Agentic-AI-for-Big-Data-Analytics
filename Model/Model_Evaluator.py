# Model_Evaluator.py
# Standalone evaluator for static saved PySpark ML models.
# Loads each saved model, evaluates on the held-out test set, and reports:
#   AUC-ROC, AUC-PRC, F1, weighted precision/recall, accuracy, confusion matrix.
# Saves ROC and PRC curve data to JSON and plots them if matplotlib is available.
# Usage: python3 Model_Evaluator.py

import json
import os
import sys
from pathlib import Path

import psutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    GBTClassificationModel,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    LinearSVCModel,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
CURVE_SAMPLE_FRACTION = 0.05   # ~50-70k rows from 1.4M — curves are approximate, AUC values are exact


# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COL    = "high_severity"
FEATURES_COL = "features"

MODEL_CLASS_MAP = {
    "org.apache.spark.ml.classification.GBTClassificationModel":          GBTClassificationModel,
    "org.apache.spark.ml.classification.LogisticRegressionModel":         LogisticRegressionModel,
    "org.apache.spark.ml.classification.RandomForestClassificationModel": RandomForestClassificationModel,
    "org.apache.spark.ml.classification.LinearSVCModel":                  LinearSVCModel,
}

# Saved model root dirs to evaluate — add more paths here as needed
MODEL_ROOTS = [
    # Path(__file__).parent / "saved_models_8combins_0.8464",
    # Path(__file__).parent / "saved_models_72combins",
    # Path(__file__).parent / "saved_models_test1combi",
    Path(__file__).parent / "model_72combi_complete",
]

PROJECT_ROOT    = Path(__file__).parent.parent
TEST_PARQUET    = PROJECT_ROOT / "dataset" / "engineered_df_test_pruned.parquet"
FE_STATE_PATH   = PROJECT_ROOT / "FE" / "state" / "fe_agent_state.json"
PLOTS_DIR       = Path(__file__).parent / "evaluation_plots"


# ── Helpers ───────────────────────────────────────────────────────────────────

def border(s):
    print()
    print(f"{'='*20} {s} {'='*20}")


def detect_model_class(model_dir: Path):
    """Read the Spark class name from model metadata and return the Python class."""
    for meta_file in (model_dir / "metadata").glob("part-*"):
        if meta_file.name.startswith("."):
            continue
        with open(meta_file) as f:
            meta = json.load(f)
        spark_class = meta.get("class", "")
        py_class    = MODEL_CLASS_MAP.get(spark_class)
        if py_class is None:
            raise ValueError(f"Unsupported model class in metadata: '{spark_class}'")
        return spark_class, py_class
    raise FileNotFoundError(f"No metadata part-file found in {model_dir / 'metadata'}")


def load_feature_cols() -> list:
    """Load feature column list from FE agent state, excluding target leakage columns."""
    with open(FE_STATE_PATH) as f:
        fe_state = json.load(f)
    return [c for c in fe_state["feature_columns"] if c != "high_severity"]


def load_feature_cols_for_model(model_dir: Path) -> list:
    """
    Return the feature list that THIS model was actually trained on, in training order.

    Priority (highest to lowest):
      1. feature_cols.json  — explicitly saved training-order list (most reliable)
      2. fe_agent_state.json — current FE run order, filtered to features in
                               feature_importances.json when that file exists.

    Note: feature_importances.json is sorted by importance (descending), NOT training
    order. Using it directly scrambles GBT tree-split indices and breaks predictions.
    """
    # ── 1. Prefer explicitly saved training order ─────────────────────────────
    fc_path = model_dir / "feature_cols.json"
    if fc_path.exists():
        with open(fc_path) as f:
            return json.load(f)

    # ── 2. Fall back to current fe_agent_state.json order ────────────────────
    base_cols = load_feature_cols()
    fi_path = model_dir / "feature_importances.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)
        fi_keys = set(fi.keys())
        filtered = [c for c in base_cols if c in fi_keys]
        if len(filtered) != len(fi_keys):
            extra = fi_keys - set(base_cols)
            if extra:
                print(f"  [warn] {len(extra)} feature(s) in feature_importances.json not in "
                      f"fe_agent_state.json (will be zero-padded): {extra}")
                filtered += sorted(extra)
        return filtered
    return base_cols


def assemble_test_df(test_df, feature_cols):
    """
    Apply the same assembly pipeline used during training:
      1. Create Severity_Binary if absent
      2. Drop stale features column if present
      3. VectorAssembler → dense UDF
    """
    if LABEL_COL not in test_df.columns:
        test_df = test_df.withColumn(LABEL_COL, when(col("Severity") >= 3, 1).otherwise(0))

    if FEATURES_COL in test_df.columns:
        test_df = test_df.drop(FEATURES_COL)

    actual_cols = set(test_df.columns)
    missing     = [c for c in feature_cols if c not in actual_cols]
    if missing:
        print(f"  [warn] Features not found in test data (padding with zeros): {missing}")
        for m in missing:
            # Pad missing scalar features with 0.0 so the assembled vector
            # has the same dimensionality as when the model was trained.
            # OHE/vector columns are stored in the parquet and won't appear here;
            # only scalar features can go missing (e.g. after pruning).
            test_df = test_df.withColumn(m, lit(0.0))

    assembler = VectorAssembler(
        inputCols=feature_cols,   # use full list — missing cols are now zero-filled
        outputCol=FEATURES_COL,
        handleInvalid="skip",
    )
    assembled = assembler.transform(test_df).select(FEATURES_COL, LABEL_COL)

    to_dense = udf(lambda v: Vectors.dense(v.toArray().tolist()), VectorUDT())
    return assembled.withColumn(FEATURES_COL, to_dense(col(FEATURES_COL)))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(predictions_df) -> dict:
    """
    Compute full evaluation suite from a predictions DataFrame.

    Returns a dict with:
      auc_roc, auc_prc,
      f1, weighted_precision, weighted_recall, accuracy,
      confusion_matrix (dict with tp/fp/tn/fn),
      roc_curve (list of {fpr, tpr}),
      prc_curve (list of {recall, precision})
    """
    # Cache predictions so the multiple evaluator passes don't recompute from scratch
    predictions_df = predictions_df.cache()
    predictions_df.count()  # materialise cache now

    # ── AUC-ROC and AUC-PRC (DataFrame API) ──────────────────────────────────
    be = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
    )
    auc_roc = round(be.evaluate(predictions_df, {be.metricName: "areaUnderROC"}), 4)
    auc_prc = round(be.evaluate(predictions_df, {be.metricName: "areaUnderPR"}),  4)

    # ── Classification metrics (DataFrame API) ────────────────────────────────
    mce = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
    )
    f1        = round(mce.evaluate(predictions_df, {mce.metricName: "f1"}),                4)
    precision = round(mce.evaluate(predictions_df, {mce.metricName: "weightedPrecision"}), 4)
    recall    = round(mce.evaluate(predictions_df, {mce.metricName: "weightedRecall"}),    4)
    accuracy  = round(mce.evaluate(predictions_df, {mce.metricName: "accuracy"}),          4)

    # ── Confusion matrix (DataFrame groupBy — stays in JVM, no Python worker) ─
    cm_rows = (
        predictions_df
        .select(
            col(LABEL_COL).cast("int").alias("actual"),
            col("prediction").cast("int").alias("predicted"),
        )
        .groupBy("actual", "predicted")
        .count()
        .collect()
    )
    cm = {(r["actual"], r["predicted"]): r["count"] for r in cm_rows}
    tp = int(cm.get((1, 1), 0))
    fp = int(cm.get((0, 1), 0))
    tn = int(cm.get((0, 0), 0))
    fn = int(cm.get((1, 0), 0))

    total = tp + fp + tn + fn
    true_precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    true_recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    true_accuracy  = round((tp + tn) / total, 4) if total > 0 else 0.0

    # ── ROC and PRC curves (driver-side, sampled) ─────────────────────────────
    # Collect a small sample to the driver and compute curves with sklearn.
    # Avoids pushing 1.4M rows through the Python RDD worker (causes OOM/crash).
    # AUC values above are already exact (computed fully in JVM); curves are approximate.
    if "probability" in predictions_df.columns:
        score_col = vector_to_array(col("probability"))[1]
    else:
        score_col = vector_to_array(col("rawPrediction"))[1]

    sample_pd = (
        predictions_df
        .select(score_col.cast("double").alias("_score"), col(LABEL_COL).cast("double"))
        .sample(fraction=CURVE_SAMPLE_FRACTION, seed=42)
        .toPandas()
    )
    scores = sample_pd["_score"].values
    labels = sample_pd[LABEL_COL].values

    predictions_df.unpersist()

    try:
        from sklearn.metrics import roc_curve as sk_roc, precision_recall_curve as sk_prc
        fpr_arr, tpr_arr, _   = sk_roc(labels, scores)
        pre_arr, rec_arr, _   = sk_prc(labels, scores)
        roc_curve = [{"fpr": round(float(f), 6), "tpr": round(float(t), 6)}
                     for f, t in zip(fpr_arr, tpr_arr)]
        prc_curve = [{"recall": round(float(r), 6), "precision": round(float(p), 6)}
                     for p, r in zip(pre_arr, rec_arr)]
    except ImportError:
        print("  [warn] sklearn not installed — curve points skipped (AUC values are still exact)")
        roc_curve, prc_curve = [], []

    return {
        "auc_roc":           auc_roc,
        "auc_prc":           auc_prc,
        "f1":                f1,
        "weighted_precision": precision,
        "weighted_recall":   recall,
        "accuracy":          accuracy,
        "true_precision":    true_precision,
        "true_recall":       true_recall,
        "true_accuracy":     true_accuracy,
        "confusion_matrix":  {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "roc_curve":         roc_curve,
        "prc_curve":         prc_curve,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_report(label: str, spark_class: str, results_json: dict, metrics: dict):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  Class : {spark_class.split('.')[-1]}")
    if results_json:
        print(f"  Saved best params : {results_json.get('best_params', {})}")
    print(f"{'─'*55}")
    print(f"  AUC-ROC                        : {metrics['auc_roc']}")
    print(f"  AUC-PRC                        : {metrics['auc_prc']}")
    print(f"  F1 (weighted)                  : {metrics['f1']}")
    print(f"  Precision  — weighted / true   : {metrics['weighted_precision']} / {metrics['true_precision']}")
    print(f"  Recall     — weighted / true   : {metrics['weighted_recall']} / {metrics['true_recall']}")
    print(f"  Accuracy   — weighted / true   : {metrics['accuracy']} / {metrics['true_accuracy']}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix (on test set):")
    print(f"                  Predicted 0   Predicted 1")
    print(f"  Actual 0            {cm['tn']:<10}  {cm['fp']}")
    print(f"  Actual 1            {cm['fn']:<10}  {cm['tp']}")
    total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
    if total > 0:
        tpr = round(cm["tp"] / max(cm["tp"] + cm["fn"], 1), 4)
        tnr = round(cm["tn"] / max(cm["tn"] + cm["fp"], 1), 4)
        print(f"\n  True Positive Rate (Sensitivity) : {tpr}")
        print(f"  True Negative Rate (Specificity) : {tnr}")


def save_curves(label: str, metrics: dict):
    """Save ROC and PRC curve points to JSON files under evaluation_plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_").replace(" ", "_")

    roc_path = PLOTS_DIR / f"{safe_label}_roc_curve.json"
    prc_path = PLOTS_DIR / f"{safe_label}_prc_curve.json"

    with open(roc_path, "w") as f:
        json.dump(metrics["roc_curve"], f, indent=2)
    with open(prc_path, "w") as f:
        json.dump(metrics["prc_curve"], f, indent=2)

    print(f"\n  ROC curve saved → {roc_path.name}")
    print(f"  PRC curve saved → {prc_path.name}")


def plot_curves(all_results: list):
    """Plot ROC and PRC curves for all models on a single figure, if matplotlib available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [info] matplotlib not installed — skipping plots (curve data saved to JSON)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Evaluation — ROC and PRC Curves", fontsize=14)

    ax_roc, ax_prc = axes

    for label, metrics in all_results:
        roc = metrics["roc_curve"]
        prc = metrics["prc_curve"]
        fpr = [p["fpr"] for p in roc]
        tpr = [p["tpr"] for p in roc]
        rec = [p["recall"]    for p in prc]
        pre = [p["precision"] for p in prc]

        auc_roc = metrics["auc_roc"]
        auc_prc = metrics["auc_prc"]

        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={auc_roc})")
        ax_prc.plot(rec, pre, label=f"{label} (AUC-PRC={auc_prc})")

    # ROC reference line
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    ax_prc.set_xlabel("Recall")
    ax_prc.set_ylabel("Precision")
    ax_prc.set_title("Precision-Recall Curve")
    ax_prc.legend(fontsize=8)
    ax_prc.grid(True, alpha=0.3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "curves_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Comparison plot saved → {plot_path}")


# ── Spark Setup ───────────────────────────────────────────────────────────────

def build_spark() -> SparkSession:
    cores          = os.cpu_count()
    total_ram_gb   = psutil.virtual_memory().total / (1024 ** 3)
    driver_mem     = f"{int(total_ram_gb * 0.7)}g"
    python_path    = sys.executable

    os.environ["PYSPARK_PYTHON"]        = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
    os.environ["SPARK_LOCAL_IP"]        = "127.0.0.1"
    os.environ["PYSPARK_PIN_THREAD"]    = "true"

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("Model_Evaluator")
        .config("spark.driver.host",               "127.0.0.1")
        .config("spark.driver.bindAddress",        "127.0.0.1")
        .config("spark.pyspark.python",            python_path)
        .config("spark.pyspark.driver.python",     python_path)
        .config("spark.driver.memory",             driver_mem)
        .config("spark.sql.shuffle.partitions",    cores * 3)
        .config("spark.default.parallelism",       cores)
        .config("spark.sql.adaptive.enabled",      "true")
        .config("spark.python.use.daemon",         "false")
        .config("spark.driver.extraJavaOptions",
                "-Djava.net.preferIPv4Stack=true "
                "-Dlog4j.logger.org.apache.spark.storage.BlockManagerStorageEndpoint=FATAL")
        .getOrCreate()
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    border("Model Evaluator — Startup")

    if not TEST_PARQUET.exists():
        raise FileNotFoundError(f"Test parquet not found: {TEST_PARQUET}")
    if not FE_STATE_PATH.exists():
        raise FileNotFoundError(f"FE state not found: {FE_STATE_PATH}")

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    border("Loading test data")
    test_df = spark.read.parquet(str(TEST_PARQUET))
    print(f"  Test rows : {test_df.count():,}")

    all_curve_results = []   # [(label, metrics), ...] for combined plot

    # Discover and evaluate every model in each root dir
    for root in MODEL_ROOTS:
        if not root.exists():
            print(f"\n  [skip] Root not found: {root}")
            continue

        model_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "model").exists()]
        if not model_dirs:
            print(f"\n  [skip] No model subdirs found in {root.name}")
            continue

        border(f"Evaluating models in: {root.name}")

        for model_dir in sorted(model_dirs):
            label = f"{root.name} / {model_dir.name}"
            print(f"\n  >> {label}")

            try:
                spark_class, py_class = detect_model_class(model_dir / "model")
                model = py_class.load(str(model_dir / "model"))
            except Exception as e:
                print(f"  [error] Could not load model: {e}")
                continue

            # Load the feature list THIS model was trained on (from feature_importances.json
            # if present, else fall back to current fe_agent_state.json).
            model_feature_cols = load_feature_cols_for_model(model_dir)
            print(f"  Feature set : {len(model_feature_cols)} cols "
                  f"({'from feature_importances.json' if (model_dir / 'feature_importances.json').exists() else 'from fe_agent_state.json'})")

            # Assemble test data using this model's specific feature set
            assembled_test = assemble_test_df(test_df, model_feature_cols)

            # Load saved results.json for best_params display
            results_json = {}
            results_file = model_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results_json = json.load(f)

            print(f"  Model loaded ({spark_class.split('.')[-1]}). Running predictions...")
            try:
                predictions = model.transform(assembled_test)

                print("  Computing metrics...")
                metrics = compute_metrics(predictions)

                print_report(label, spark_class, results_json, metrics)
                save_curves(label, metrics)

                all_curve_results.append((f"{root.name}/{model_dir.name}", metrics))
            except Exception as e:
                # Most likely cause: model was trained on a different feature set
                # (vector size mismatch). Skip and continue to the next model.
                print(f"  [error] Evaluation failed — model may be stale (trained on different features): {e}")

    # Combined comparison plot
    if all_curve_results:
        border("Generating comparison plots")
        plot_curves(all_curve_results)

    # Summary table
    border("Summary")
    header = (
        f"{'Model':<50}  {'AUC-ROC':>8}  {'AUC-PRC':>8}  {'F1':>8}"
        f"  {'Prec(wtd)':>10}  {'Prec(true)':>10}"
        f"  {'Rec(wtd)':>8}  {'Rec(true)':>9}"
        f"  {'Acc(wtd)':>8}  {'Acc(true)':>9}"
    )
    print(header)
    print("─" * len(header))
    for label, metrics in all_curve_results:
        print(
            f"{label:<50}  "
            f"{metrics['auc_roc']:>8}  "
            f"{metrics['auc_prc']:>8}  "
            f"{metrics['f1']:>8}  "
            f"{metrics['weighted_precision']:>10}  "
            f"{metrics['true_precision']:>10}  "
            f"{metrics['weighted_recall']:>8}  "
            f"{metrics['true_recall']:>9}  "
            f"{metrics['accuracy']:>8}  "
            f"{metrics['true_accuracy']:>9}"
        )

    spark.stop()
    border("Done")


if __name__ == "__main__":
    main()
