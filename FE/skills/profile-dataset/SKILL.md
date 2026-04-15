---
name: validate-cleaning
description: >
  Comprehensively profile a PySpark DataFrame to surface data quality issues.
  Produces a structured JSON-like profile used by the downstream feature engineering
  agent to automate imputation, filtering, and feature selection.
mode: organisational
inputs:
  - df: PySpark DataFrame
  - target_col: Name of the target variable
  - timestamp_cols: List of columns to treat as timestamps
outputs:
  - profile: Nested dictionary of all data characteristics and recommendations
---

# Dataset Profiling Skill

## Purpose
This skill is the FIRST node in the data pipeline. It performs a thorough
examination of the input DataFrame and produces a profile dictionary.
This profile is the input to the downstream feature engineering agent.

## Key Metrics & Logic
- **Structural:** Total rows, columns, column names, and data types.
- **Missingness:** Categorizes null density from "none" to "critical."
- **Numerical:** Uses IQR for outlier detection and Skewness for imputation strategy (Mean vs Median).
- **Categorical:** Flags high cardinality and "Near-Zero Variance" (top category > 95%).
- **Temporal:** Detects "Sparse Years" (years with < 1% of total data) to identify reporting shifts.
- **Correlation:** Computes Pearson correlation to identify redundant features (Threshold > 0.9).

## Output Format
Return a single Python dictionary with this exact structure:
```python
profile = {
    "structural_overview": {
        "total_rows": int,
        "total_columns": int,
        "column_types": {
            "col_name": "numerical/categorical/timestamp/time_derived/boolean"
        }
    },
    "missing_values": {
        "col_name": {
            "null_count": int,
            "null_pct": float,
            "severity": "none/low/moderate/high/critical",
            "recommendation": str
        }
    },
    "numerical_analysis": {
        "col_name": {
            "mean": float,
            "std": float,
            "min": float,
            "max": float,
            "skewness": float,
            "kurtosis": float,
            "outlier_count": int,
            "distribution": "normal/moderate_skew/high_skew",
            "imputation_recommendation": "mean/median"
        }
    },
    "categorical_analysis": {
        "col_name": {
            "distinct_count": int,
            "top_5_values": {str: int},
            "bottom_5_values": {str: int},
            "near_zero_variance": bool,
            "encoding_recommendation": str
        }
    },
    "timestamp_analysis": {
        "col_name": {
            "min": str,
            "max": str,
            "yearly_distribution": {str: int},
            "sparse_years": [int],
            "temporal_consistency": "consistent/inconsistent"
        }
    },
    "target_analysis": {
        "column": str,
        "class_distribution": {str: int},
        "imbalance_ratio": float,
        "imbalance_classification": str,
        "smote_recommendation": str
    },
    "correlation_analysis": {
        "highly_correlated_pairs": [
            {
                "col1": str,
                "col2": str,
                "correlation": float,
                "recommendation": str
            }
        ],
        "near_duplicate_pairs": [
            {
                "col1": str,
                "col2": str,
                "correlation": float,
                "recommendation": str
            }
        ]
    },
    "quality_summary": {
        "total_issues": int,
        "critical_issues": [str],
        "warnings": [str],
        "overall_quality_score": "good/moderate/poor"
    }
}
```

## Important Notes
- This skill is PURE PYTHON AND PYSPARK — no LLM is involved
- All computations must be done using PySpark functions
- Cache the DataFrame before profiling to avoid recomputation:
  df.cache()
- The output profile is stored in AgentState["profile"]
- The downstream cleaning agent reads AgentState["profile"]
  and uses it as its SOLE INPUT for making cleaning decisions
- Do not hardcode any column names — profile ALL columns
  dynamically based on schema
- Log progress as each section completes so the user can
  track profiling progress on large datasets