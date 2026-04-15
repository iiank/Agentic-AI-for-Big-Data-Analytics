# Modelling_Skills.py
# Pure PySpark MLlib execution layer for the Validate Cleaning agent.

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict, Literal
from getpass import getpass

from pydantic import BaseModel, Field
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from pathlib import Path
import psutil

from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, year, month, dayofweek, hour, when,
    unix_timestamp, skewness, kurtosis,
    percentile_approx, count, isnan
)
from pyspark.sql import types as T
from pyspark.sql.types import StructType, StructField, IntegerType 

from pyspark.storagelevel import StorageLevel


# ── Orchestration Function ─────────────────────────────────────────────────────────

def validate_cleaning(df, target_col, timestamp_cols=None):
    """
    Coordinator function that profiles a PySpark DataFrame.
    Produces a structured dictionary consumed by the downstream feature
    engineering agent.
    """
    timestamp_cols = timestamp_cols or []
    profile = {}

    # Initial setup
    df.cache()
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"Profiling {total_rows:,} rows x {total_cols} columns...")

    # Sections as described in skill.md
    profile["structural_overview"], numerical_cols, categorical_cols, time_derived_cols = _get_structural_overview(df, total_rows, timestamp_cols)
    print("✓ Section 1: Structural overview complete")

    profile["missing_values"] = _profile_missing_values(df, total_rows)
    print("✓ Section 2: Missing value analysis complete")

    profile["numerical_analysis"] = _profile_numerical_columns(df, numerical_cols, profile["missing_values"])
    print("✓ Section 3: Numerical column analysis complete")

    profile["categorical_analysis"] = _profile_categorical_columns(df, categorical_cols, total_rows)
    print("✓ Section 4: Categorical column analysis complete")

    profile["timestamp_analysis"] = _profile_timestamps(df, timestamp_cols, total_rows)
    print("✓ Section 5: Timestamp analysis complete")

    profile["target_analysis"] = _profile_target(df, target_col, total_rows)
    print("✓ Section 6: Target column analysis complete")

    profile["correlation_analysis"] = _profile_correlations(df, numerical_cols)
    print("✓ Section 7: Correlation analysis complete")

    cleaning_rec, feature_eng_rec = _generate_agent_recommendations(profile)
    print("✓ Section 8: Generate agent recommendations complete")

    # Quality Summary
    profile["quality_summary"] = _generate_quality_summary(profile)
    print(f"✓ Profile complete. Quality Score: {profile['quality_summary']['overall_quality_score'].upper()}")

    df.unpersist()
    return {
        "profile": profile,
        "cleaning_rec": cleaning_rec,
        "feature_eng_rec": feature_eng_rec,
        "quality_score": profile["quality_summary"]["overall_quality_score"]
    }


def validate_test_split(df_test, train_columns, timestamp_cols=None):
    """
    Build a separate test profile without influencing train-driven decisions.
    """
    timestamp_cols = timestamp_cols or []

    train_cols = set(train_columns)
    test_cols = set(df_test.columns)
    missing_in_test = sorted(train_cols - test_cols)
    extra_in_test = sorted(test_cols - train_cols)

    if missing_in_test or extra_in_test:
        raise ValueError(
            "Train/Test schema mismatch detected. "
            f"Missing in test: {missing_in_test[:10]} | Extra in test: {extra_in_test[:10]}"
        )

    total_rows = df_test.count()
    structural_overview, _, _, _ = _get_structural_overview(df_test, total_rows, timestamp_cols)
    missing_values = _profile_missing_values(df_test, total_rows)

    return {
        "profile": {
            "structural_overview": structural_overview,
            "missing_values": missing_values,
        },
        "quality_score": "schema_validated",
        "schema_check": {
            "status": "pass",
            "missing_in_test": missing_in_test,
            "extra_in_test": extra_in_test,
        },
    }

# ── Dataset Profiling Helper Functions ─────────────────────────────────────────────────────────

def _is_time_derived_column(col_name, timestamp_cols):
    for col in timestamp_cols:
        if col_name.startswith(f"{col}_"):
            return True

    return False

def _get_structural_overview(df, total_rows, timestamp_cols):
    col_types = {}
    numerical_cols = []
    categorical_cols = []
    time_derived_cols = []

    for field in df.schema.fields:
        dtype = str(field.dataType)
        name = field.name

        if "TimestampType" in dtype or "DateType" in dtype or name in timestamp_cols:
            col_types[name] = "timestamp"
        elif _is_time_derived_column(name, timestamp_cols):
            col_types[name] = "time_derived"
            time_derived_cols.append(name)
        elif "BooleanType" in dtype:
            col_types[name] = "boolean"
        elif any(t in dtype for t in ["IntegerType", "LongType", "DoubleType", "FloatType"]):
            col_types[name] = "numerical"
            numerical_cols.append(name)
        else:
            col_types[name] = "categorical"
            categorical_cols.append(name)

    overview = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "column_types": col_types
    }
    return overview, numerical_cols, categorical_cols, time_derived_cols

def _profile_missing_values(df, total_rows):
    null_exprs = []

    for field in df.schema:
        name = field.name
        dtype = field.dataType

        condition = F.col(name).isNull()
        if isinstance(dtype, (T.DoubleType, T.FloatType)):
            condition = condition | F.isnan(F.col(name))

        null_exprs.append(F.count(F.when(condition, name)).alias(name))

    null_counts = df.select(null_exprs).collect()[0].asDict()

    missing_analysis = {}
    recs = {
        "none": "keep",
        "low": "impute/drop rows",
        "moderate": "impute",
        "high": "check relevance",
        "critical": "drop"
    }

    for c in df.columns:
        n = null_counts[c]
        pct = round((n / total_rows) * 100, 4)

        if pct == 0: severity = "none"
        elif pct <= 5: severity = "low"
        elif pct <= 20: severity = "moderate"
        elif pct <= 50: severity = "high"
        else: severity = "critical"

        missing_analysis[c] = {
            "null_count": n,
            "null_pct": pct,
            "severity": severity,
            "recommendation": recs[severity]
        }

    return missing_analysis

def _profile_numerical_columns(df, columns, missing_values):
    results = {}

    stats_exprs = []
    for c in columns:
        stats_exprs.extend([
            F.mean(c).alias(f"{c}_mean"),
            F.stddev(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
            F.percentile_approx(c, [0.25, 0.75]).alias(f"{c}_q"),
            F.skewness(c).alias(f"{c}_skew"),
            F.kurtosis(c).alias(f"{c}_kurt")
        ])

    # single pass agg
    all_stats = df.select(stats_exprs).collect()[0].asDict()

    outlier_exprs = []
    processed_stats = {}

    for c in columns:
        m = all_stats[f"{c}_mean"]
        s = all_stats[f"{c}_std"]
        mn = all_stats[f"{c}_min"]
        mx = all_stats[f"{c}_max"]
        q_list = all_stats[f"{c}_q"]
        sk = all_stats[f"{c}_skew"]
        kt = all_stats[f"{c}_kurt"]

        q25, q75 = q_list[0], q_list[1]
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # count outliers without a filter
        if iqr == 0:
            outlier_exprs.append(F.lit(0).alias(f"{c}_outliers"))
        else:
            outlier_exprs.append(
                F.count(F.when((F.col(c) < lower_bound) | (F.col(c) > upper_bound), c)).alias(f"{c}_outliers")
            )

        processed_stats[c] = {
            "m": m, "s": s, "min": mn, "max": mx, "sk": sk, "kt": kt, "iqr": iqr
        }

    # single pass for outliers
    all_outliers = df.select(outlier_exprs).collect()[0].asDict()

    for c in columns:
        stats = processed_stats[c]
        skew_val = abs(stats["sk"] or 0)

        if skew_val <= 0.5:
            distribution = "normal"
            impute_rec = "mean"
        elif skew_val <= 1.0:
            distribution = "moderate_skew"
            impute_rec = "median"
        else:
            distribution = "high_skew"
            impute_rec = "median"

        # if no missing values,rec = not needed
        if missing_values[c]["null_count"] == 0:
            impute_rec = "not needed"

        results[c] = {
            "mean": round(stats["m"] or 0, 4),
            "std": round(stats["s"] or 0, 4),
            "min": round(stats["min"] or 0, 4),
            "max": round(stats["max"] or 0, 4),
            "skewness": round(stats["sk"] or 0, 4),
            "kurtosis": round(stats["kt"] or 0, 4),
            "outlier_count": all_outliers[f"{c}_outliers"],
            "distribution": distribution,
            "imputation_recommendation": impute_rec
        }

    return results

def _profile_categorical_columns(df, columns, total_rows):
    results = {}

    # Columns known to contain repeating semantic sub-tokens
    discovery_targets = ["Street", "Weather_Condition"]
    stop_words = {"", "n", "s", "e", "w", "and", "the", "with", "near", "at", "by", "of", "in", "to", "for", "from", "mostly", "partly", "with"}

    for c in columns:
        # Standard profiling
        distinct_cnt = df.select(c).distinct().count()
        top_values = df.groupBy(c).count().orderBy(F.desc("count")).limit(5).collect()
        top_dict = {str(r[c]): r["count"] for r in top_values}
        top_count = list(top_dict.values())[0] if top_dict else 0

        # Automated token discovery
        extraction_hint = None

        if c in discovery_targets:
            tokens_df = df.select(F.explode(F.split(F.lower(F.col(c)), r"[\s,/-]+")).alias("token")) \
                          .withColumn("token", F.regexp_replace(F.col("token"), r"[^a-z0-9]", ""))

            discovered_tokens = tokens_df.groupBy("token").count() \
                                         .orderBy(F.desc("count")) \
                                         .limit(20) \
                                         .collect()
            keywords = [
                r["token"] for r in discovered_tokens
                if r["token"] not in stop_words and not r["token"].isdigit()
            ][:10]

            extraction_hint = {
                "type": "string_keyword_extraction",
                "discovered_patterns": keywords,
                "coverage_pct": round((top_count / total_rows) * 100, 2)
            }

        if distinct_cnt <= 10:
            rec = "one_hot_encode"
        elif extraction_hint:
            rec = "extract_binary_features"
        else:
            rec = "group_rare_categories_and_label_encode"

        results[c] = {
            "distinct_count": distinct_cnt,
            "top_5_values": top_dict,
            "near_zero_variance": (top_count / total_rows) > 0.95,
            "extraction_hint": extraction_hint,
            "encoding_recommendation": rec
        }

    return results

def _profile_timestamps(df, columns, total_rows):
    results = {}
    for c in columns:
        temp_df = df.withColumn("_ts", F.to_timestamp(col(c)))

        stats = temp_df.select(
            F.min("_ts").alias("min_ts"),
            F.max("_ts").alias("max_ts")
        ).collect()[0]

        yearly = temp_df.withColumn("year", F.year("_ts")).groupBy("year").count().collect()
        yearly_dist = {str(r["year"]): r["count"] for r in yearly}

        # sparse = < 1% total rows
        sparse = [yr for yr, count in yearly_dist.items() if count < (total_rows * 0.01)]

        results[c] = {
            "min": str(stats["min_ts"]),
            "max": str(stats["max_ts"]),
            "yearly_distribution": yearly_dist,
            "sparse_years": [int(yr) for yr in sparse],
            "temporal_consistency": "inconsistent" if sparse else "consistent"
        }
    return results

def _profile_target(df, target_col, total_rows):
    dist = df.groupBy(target_col).count().collect()
    counts = {str(r[target_col]): r["count"] for r in dist}

    class_pcts = {k: round(v/total_rows*100, 4) for k, v in counts.items()}

    vals = list(counts.values())
    imbalance_ratio = max(vals) / min(vals) if min(vals) > 0 else 0

    if imbalance_ratio > 10:
        classification = "severe_imbalance"
        smote_rec = "strongly recommend SMOTE"
    elif imbalance_ratio > 3:
        classification = "moderate_imbalance"
        smote_rec = "consider SMOTE"
    else:
        classification = "balanced"
        smote_rec = "not needed"

    return {
        "column": target_col,
        "class_distribution": counts,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "imbalance_classification": classification,
        "smote_recommendation": smote_rec
    }

def _profile_correlations(df, numerical_cols):
    high_corr_pairs = []
    near_duplicate_pairs = []
    # limit to 20 if many numerical cols
    cols_to_check = numerical_cols[:20]

    for i in range(len(cols_to_check)):
        for j in range(i + 1, len(cols_to_check)):
            c1, c2 = cols_to_check[i], cols_to_check[j]
            correlation = df.stat.corr(c1, c2)

            if abs(correlation) > 0.95:
                near_duplicate_pairs.append({
                    "col1": c1,
                    "col2": c2,
                    "correlation": round(correlation, 4),
                    "recommendation": "drop one"
                })
            elif abs(correlation) > 0.85:
                high_corr_pairs.append({
                    "col1": c1,
                    "col2": c2,
                    "correlation": round(correlation, 4),
                    "recommendation": "drop one" if abs(correlation) > 0.95 else "feature reduction"
                })
    return {"highly_correlated_pairs": high_corr_pairs, "near_duplicate_pairs": near_duplicate_pairs}

def _generate_quality_summary(profile):
    critical_issues = []
    warnings = []

    # Check for critical nulls
    for c, data in profile["missing_values"].items():
        if data["severity"] == "critical":
            critical_issues.append(f"Critical Nulls: {c}")

    # Determine score
    if not critical_issues and len(warnings) < 5:
      score = "good"
    elif len(critical_issues) <= 3 or len(warnings) <= 10:
      score = "moderate"
    else:
      score = "poor"

    return {
        "total_issues": len(critical_issues) + len(warnings),
        "critical_issues": critical_issues,
        "warnings": warnings,
        "overall_quality_score": score
    }

def _generate_agent_recommendations(profile):
    # flatten nested dicts
    cleaning_rec = []
    feature_eng_rec = []

    for col, data in profile["missing_values"].items():
        if data["severity"] == "critical":
            cleaning_rec.append({"action": "drop_column", "column": col, "reason": "critical_null_density"})
        elif data["severity"] in ["low", "moderate"]:
            cleaning_rec.append({
                "action": "impute",
                "column": col,
                "strategy": profile["numerical_analysis"].get(col, {}).get("imputation_recommendation", "mode")
            })

    for col, data in profile["categorical_analysis"].items():
        if data.get("extraction_hint"):
            feature_eng_rec.append({
                "action": "extract_binary_features",
                "column": col,
                "logic": data["extraction_hint"]["type"],
                "keywords": data["extraction_hint"]["discovered_patterns"],
                "reason": "High cardinality with repeating semantic tokens"
            })
        # else:
        #     distinct_count=data.get("distinct_count", 0)
        #     feature_eng_rec.append({
        #         "action": "encode",
        #         "column": col,
        #         "method": data["encoding_recommendation"],
        #         "reason": f"Column has {distinct_count} unique values"
        #     })
        elif not data.get("zero_variance") and not data.get("likely_identifier"):
            feature_eng_rec.append({
                "action": "encode",
                "column": col,
                "method": data["encoding_recommendation"],
                "reason": f"Column has {data.get('distinct_count', 0)} unique values"
            })

    for pair in profile["correlation_analysis"].get("near_duplicate_pairs", []):
        cleaning_rec.append({"action": "drop_column", "column": pair["col2"], "reason": f"redundant_with_{pair['col1']}"})

    # for col, data in profile["categorical_analysis"].items():
    #     if not data.get("zero_variance") and not data.get("likely_identifier"):
    #         feature_eng_rec.append({"action": "encode", "column": col, "method": data["encoding_recommendation"]})

    target = profile["target_analysis"]
    if target["imbalance_classification"] in ["severe_imbalance", "moderate_imbalance"]:
        feature_eng_rec.append({"action": "balance_target", "method": "SMOTE", "ratio": target["imbalance_ratio"]})

    return cleaning_rec, feature_eng_rec
