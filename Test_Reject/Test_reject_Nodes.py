import json
import os
import numpy as np
from getpass import getpass
from typing import TypedDict, Optional, Literal, Any, Dict, List
from pydantic import BaseModel, Field
from openai import OpenAI

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col, year, month, dayofweek, hour, when,
    unix_timestamp, skewness, kurtosis,
    percentile_approx, count, isnan
)

from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer, VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline

from pyspark.storagelevel import StorageLevel

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from pathlib import Path
import psutil

from IPython.display import Image, display
from dotenv import load_dotenv

from ValidateCleaning_Skills import validate_cleaning, validate_test_split

class AgentState(TypedDict):
    # --- Phase 1: Cleaning Validation ---
    profile_train: Optional[dict]           # train profile consumed by FE
    profile_test: Optional[dict]            # separate test profile/validation
    cleaning_plan: Optional[dict]           # optional future cleaning plan
    validation_status: Optional[str]        # approved | rejected | stopped
    validation_message: Optional[str]       # details for final decision
    auto_decision: Optional[dict]           # accept/reject payload from cleaning gate
    execution_log: list                     # step-by-step trace

    # --- Phase 2: Feature Engineering ---
    semantic_decisions: dict
    numeric_decisions: dict
    categorical_decisions: dict
    engineered_df_train_path: str           # Path to post-FE parquet
    engineered_df_test_path: str
    feature_cols: List[str]                 # Final assembler cols list
    feature_validation_report: dict         # Structural checks (Level 1)
    feature_stats_report: dict              # Statistical profile (Level 2)
    validation_iterations: int
    validation_passed: bool
    validation_failures: List[str]
    #ouptut from feature gate
    feature_auto_decision: Optional[dict]
    feature_validation_status: Optional[dict]
    feature_validation_message: Optional[str]
    cols_to_prune: Optional[List[str]]

# Direct post-profiling gate: decide whether cleaning is acceptable.
def cleaning_accept_reject_node(state: AgentState) -> AgentState:
    train_profile_bundle = state.get("profile_train") or {}
    nested_profile = train_profile_bundle.get("profile") or {}
    quality_summary = nested_profile.get("quality_summary") or {}

    raw_quality_score = train_profile_bundle.get("quality_score")
    quality_band = str(raw_quality_score or "").strip().lower()

    # Support both quality_score formats:
    # 1) numeric score (0-1)
    # 2) categorical band (good/moderate/poor)
    quality_value = None
    if isinstance(raw_quality_score, (int, float)):
        quality_value = max(0.0, min(1.0, float(raw_quality_score)))
    else:
        quality_value = {
            "good": 1.0,
            "moderate": 0.7,
            "poor": 0.2,
        }.get(quality_band, 0.0)

    total_issues = int(quality_summary.get("total_issues", 0) or 0)

    critical_issues = quality_summary.get("critical_issues", [])
    if not isinstance(critical_issues, list):
        critical_issues = []
    critical_issue_count = len(critical_issues)

    # Accept when critical issues are absent and quality passes baseline.
    accepted = (critical_issue_count == 0) and (quality_value >= 0.70)

    decision = "accept" if accepted else "reject"
    if accepted:
        validation_status = "approved"
        reason = (
            f"Accepted for feature engineering. quality_score={raw_quality_score}, "
            f"critical_issues={critical_issue_count}, total_issues={total_issues}."
        )
    else:
        validation_status = "rejected"
        reason = (
            f"Rejected after profiling. quality_score={raw_quality_score}, "
            f"critical_issues={critical_issue_count}, total_issues={total_issues}."
        )

    auto_decision = {
        "decision": decision,
        "quality_score": raw_quality_score,
        "quality_value": round(quality_value, 4),
        "critical_issue_count": critical_issue_count,
        "total_issues": total_issues,
        "reason": reason,
    }

    state["auto_decision"] = auto_decision
    state["validation_status"] = validation_status
    state["validation_message"] = reason
    state.setdefault("execution_log", []).append({"node": "cleaning_gate", **auto_decision})

    print("=" * 55)
    print(">> NODE 2: Cleaning accept/reject gate")
    print("=" * 55)
    print(json.dumps(auto_decision, indent=2))
    return state

def route_decision(state: AgentState) -> str:
    auto_decision = state.get("auto_decision") or {}
    if auto_decision.get("decision") == "accept":
        return "semantic_agent"
    return "end"

def create_secure_openai_client():
    """
    Create OpenAI client with secure API key handling.

    This function:
    1. Looks for OPENAI_API_KEY in environment variables
    2. Tests the connection with a simple API call
    3. Returns the client or None if setup fails
    """
    try:
        load_dotenv()  # Load .env file if it exists
    except ImportError:
        pass  # python-dotenv not installed, that's okay

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OpenAI API key found.")
        print("Set environment variable: OPENAI_API_KEY=your_key")
        print("Or create .env file with: OPENAI_API_KEY=your_key")
        return None

    try:
        client = OpenAI(api_key=api_key)
        # Test connection with a simple API call
        models = client.models.list()
        print("OpenAI client created and tested successfully")
        return client
    except Exception as e:
        print(f"OpenAI client creation failed: {e}")
        print("Check your API key and internet connection")
        return None

def load_skill(filename: str) -> str:
    """Read a skill.md file from disk (supports both SKILLS/ and skills/)."""
    path = os.path.join("SKILLS", filename)
    with open(path, "r") as f:
        return f.read()

    raise FileNotFoundError(f"Skill file not found in SKILLS/ or skills/: {filename}")

def semantic_agent_node(state: AgentState) -> AgentState:
    all_recs = state["profile_train"].get("feature_eng_rec", [])
    semantic_plan = [rec for rec in all_recs if rec.get('action') == "extract_binary_features"]
    context_str = json.dumps({"semantic_tasks": semantic_plan}, indent=2)
    skill_doc = load_skill("skill_semantic_boolean_expansion.md")

    #retrieve feedback from validate feature eng gate on semantic boolean expansions
    iteration = state.get("validation_iterations",0)
    feedback_prompt = ""
    if iteration > 0 and state.get("feature_auto_decision"):
        prev_decision = state["feature_auto_decision"]
        bad_flags = prev_decision.get("suspicious_semantic_flags", [])
        
        if bad_flags:
            feedback_prompt = "\n===Feedback from previous failed attempt===\n"
            feedback_prompt += "Your previous regex patterns resulted in features with a true_ratio that was too low (rare) or too high (overly general). Do not use these same mappings again:\n"
            for flag in bad_flags:
                feedback_prompt += f"- Target {flag['target_column']} | Reason: {flag['reason']}\n"
            feedback_prompt += "Action: Broaden the regex by combining terms, or choose different more semnatic categories.\n"
    system_prompt=f"""You are the semantic feature engineer for a PySpark Pipeline predicting US accident severity.
    Your only job is to look at compound text columns and extract boolean flags using regex.

    ===SKILL MANUAL===
    {skill_doc}

    CRITICAL RULES:
    - EXHAUSTIVE COVERAGE: you MUST generate at least 10 distinct flags for Weather_Condition (eg. is_rain, is_snow, is_fog...)
    - You must generate at least 10 distinct flags for Street (eg. is_highway, is_local_street, is_country_road)
    - Write regex strictly in lowercase
"""
    user_prompt=f"Data profile:\n{context_str}\n"
    if feedback_prompt:
        user_prompt += feedback_prompt
    user_prompt += "\n Generate semantic expansions using the discovered keywords."
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=SemanticDecisionList
    )

    #ouput logs for tracking agent behaviour
    decision = response.choices[0].message.parsed
    print("\n" + "="*60)
    print("SEMANTIC AGENT DECISIONS")
    print("="*60)
    for exp in decision.semantic_expansions:
        print(f"\n{exp.source_column:<18} -> {exp.target_column:<20} (Regex: {exp.regex_pattern})")
        print(f"Rationale: {exp.rationale}")
    print(f"\nOverall Rationale: {decision.overall_rationale}")
    print("="*60 + "\n")


    return {"semantic_decisions": json.loads(decision.model_dump_json())}

def numeric_agent_node(state: AgentState) -> AgentState:
    data_profile = state["profile_train"]["profile"]
    sliced_context = {
        "numerical_columns": data_profile.get("numerical_analysis",{}),
        "timestamp_columns": list(data_profile.get("timestamp_analysis", {}).keys()),
        "total_rows": data_profile.get("structural_overview").get("total_rows")
    }
    context_str = json.dumps(sliced_context, indent=2)
    skills = "\n---\n".join([
        load_skill("skill_extract_time_features.md"),
        load_skill("skill_bin_numeric.md"),
        load_skill("skill_compute_interaction_features.md"),
        load_skill("skill_scale_numeric.md")
    ])
    
    system_prompt = f"""You are the Numeric Feature Engineer for a PySpark pipeline predicting US accident severity.
Your job is to handle timestamps, numeric binning, numeric interactions, and continuous scaling.

=== SKILL MANUALS ===
{skills}

CRITICAL RULES:
- BINNING: You MUST apply 'skill_bin_numeric' to discretize at least 'Temperature(F)' and 'Visibility(mi)'. Use 'custom_cuts'.
- INTERACTIONS: You MUST invent at least 5 novel, logical interaction feature using multiply, add, subtract or ratio. 
    Domain knowldege: Think like a traffic safety engineer. Look for combinations that amplify driving danger:
    - Severe weather: compounding effects (eg. Wind_Speed * Precipitation implies a rain storm)
    - Freezing risks: interactions between Temperature and precipitation or humidity
    - Kinematics: Rates of travels, such as distance / duration (a proxy for speed or traffic flow)

"""
    user_prompt = f"Data Profile:\n{context_str}\n\nGenerate the numeric transformations."
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=NumericDecisionList,
    )

    #ouput logs for tracking agent behaviour
    decision = response.choices[0].message.parsed
    print("\n" + "="*60)
    print("NUMERIC AGENT DECISIONS")
    print("="*60)
    if decision.time_features:
        print(f"\nTime Features: {decision.time_features.features_to_extract}")
        print(f"Rationale: {decision.time_features.rationale}")
    if decision.bins:
        for b in decision.bins:
            print(f"\nBinning: {b.column} ({b.strategy})")
            print(f"Rationale: {b.rationale}")
    if decision.interactions:
        for ix in decision.interactions:
            print(f"\nInteraction: {ix.col_a} {ix.op} {ix.col_b}")
            print(f"Rationale: {ix.rationale}")

    print(f"\nScaling Applied: {decision.apply_scaling} (Strategy: {decision.scaling_strategy})")
    if decision.scaling_decisions:
        for s in decision.scaling_decisions:
            print(f"- {s.column}: {s.action} | {s.rationale}")
    if decision.scaling_rationale:
        print(f"\nRationale: {decision.scaling_rationale}")
        
    print(f"\nOverall Rationale: {decision.overall_rationale}")
    print("="*60 + "\n")

    return {"numeric_decisions": json.loads(decision.model_dump_json())}
    
def categorical_agent_node(state: AgentState) -> AgentState:
    all_recs = state["profile_train"].get("feature_eng_rec", [])
    profile_data = state["profile_train"]["profile"]
    encoding_plan = [rec for rec in all_recs if rec.get("action") == "encode"]
    timestamps = list(profile_data.get("timestamp_analysis", {}).keys())
    sliced_context = {
        "encoding_tasks": encoding_plan,
        "timestamps_to_drop": timestamps
    }
    context_str = json.dumps(sliced_context, indent=2)
    skills = "\n---\n".join([
        load_skill("skill_encode_categorical.md"),
        load_skill("skill_drop_columns.md")
    ])
    
    system_prompt = f"""You are the Categorical Feature Engineer for a PySpark pipeline predicting US accident severity.
Your job is to encode the remaining string columns and designate which original columns to drop.

=== SKILL MANUALS ===
{skills}

CRITICAL RULES:
- For high cardinality columns (>20 unique values) like City, State, or Zipcode, you MUST use 'string_index_only' to prevent OOM errors.
- Always include 'Start_Time' and original encoded string columns in 'drop_after_encoding'.
- For every column in 'drop_after_encoding', you must provide a clear 'rationale' for dropping the column (eg. 'Dropped because it was one hot encoded', 'Dropped because temporal features were extracted')
"""
    user_prompt = f"Data Profile:\n{context_str}\n\nGenerate the categorical encodings and drop list."
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=CategoricalDecisionList,
    )
    
    decision = response.choices[0].message.parsed
    print("\n" + "="*60)
    print("CATEGORICAL AGENT DECISIONS")
    print("="*60)
    for enc in decision.column_encodings:
        print(f"Encode: {enc.column:<22} -> {enc.action}")
        print(f" | Rationale: {enc.rationale}")
        
    print(f"\nColumns to Drop:")
    for drop in decision.drop_after_encoding:
        print(f" - Drop: {drop.column: <22} | Rationale: {drop.rationale}")
        
    print(f"\nOverall Rationale: {decision.overall_rationale}")
    print("="*60 + "\n")

    return {"categorical_decisions": json.loads(decision.model_dump_json())}

def execute_feature_eng(state: AgentState):
    train_data = train_df
    test_data = test_df
    
    # sub-agent decisions
    sem = SemanticDecisionList.model_validate(state["semantic_decisions"])
    num = NumericDecisionList.model_validate(state["numeric_decisions"])
    cat = CategoricalDecisionList.model_validate(state["categorical_decisions"])
    
    assembler_cols = []
    
    # operations applied to both train and test sets
    def apply_stateless_features(df):
        local_assembler_cols = []
        
        # time features
        if num.time_features:
            for feat in num.time_features.features_to_extract:
                if feat == "is_rush_hour":
                    df = df.withColumn("is_rush_hour", F.when(F.hour("Start_Time").between(7,9) | F.hour("Start_Time").between(17,19), 1).otherwise(0))
                    if "is_rush_hour" not in local_assembler_cols: local_assembler_cols.append("is_rush_hour")

        # semantic boolean expansions
        for exp in sem.semantic_expansions:
            if exp.source_column in df.columns:
                lower_col = F.lower(F.col(exp.source_column))
                df = df.withColumn(exp.target_column, lower_col.rlike(exp.regex_pattern).cast("integer")).fillna(0, subset=[exp.target_column])
                if exp.target_column not in local_assembler_cols: local_assembler_cols.append(exp.target_column)

        # numeric interactions
        ops = {
            "multiply": lambda a, b: F.col(a) * F.col(b), 
            "add": lambda a, b: F.col(a) + F.col(b), 
            "subtract": lambda a, b: F.col(a) - F.col(b),
            "ratio": lambda a, b: F.col(a) / (F.col(b) + F.lit(1e-6))
        }
        if num.interactions:
            for ix in num.interactions:
                if ix.col_a in df.columns and ix.col_b in df.columns:
                    out_col = f"{ix.col_a}_{ix.op}_{ix.col_b}"
                    df = df.withColumn(out_col, ops[ix.op](ix.col_a, ix.col_b))
                    if out_col not in local_assembler_cols: local_assembler_cols.append(out_col)

        # manual boolean encoding
        for field in df.schema.fields:
            if isinstance(field.dataType, T.BooleanType) and field.name != "Severity":
                df = df.withColumn(field.name, F.col(field.name).cast("integer"))
                if field.name not in local_assembler_cols: local_assembler_cols.append(field.name)
                
        return df, local_assembler_cols

    # Execute stateless operations on both train and test
    train_data, stateless_cols = apply_stateless_features(train_data)
    test_data, _ = apply_stateless_features(test_data)
    
    assembler_cols.extend(stateless_cols)

    # operations applied to train data only
    stages = []

    # numeric binning, find and apply splits based on training data only
    if num.bins:
        for b in num.bins:
            if b.column in train_data.columns:
                if b.strategy == "equal_width":
                    mn = train_data.agg({b.column: "min"}).collect()[0][0]
                    mx = train_data.agg({b.column: "max"}).collect()[0][0]
                    splits = [float(x) for x in np.linspace(mn, mx, b.n_bins + 1)]
                    splits[0], splits[-1] = float("-inf"), float("inf")
                else:
                    splits = [float("-inf")] + [float(c) for c in b.cuts] + [float("inf")]
                
                out_col = f"{b.column}_bin"
                bucketizer = Bucketizer(splits=splits, inputCol=b.column, outputCol=out_col, handleInvalid="keep")
                stages.append(bucketizer)
                assembler_cols.append(out_col)

    # categorical encoding
    for enc in cat.column_encodings:
        if enc.column in train_data.columns:
            if enc.action == "string_index_ohe":
                stages.append(StringIndexer(inputCol=enc.column, outputCol=f"{enc.column}_idx", handleInvalid="keep"))
                stages.append(OneHotEncoder(inputCol=f"{enc.column}_idx", outputCol=f"{enc.column}_ohe"))
                assembler_cols.append(f"{enc.column}_ohe")
            elif enc.action == "string_index_only":
                stages.append(StringIndexer(inputCol=enc.column, outputCol=f"{enc.column}_idx", handleInvalid="keep"))
                assembler_cols.append(f"{enc.column}_idx")
            elif enc.action == "passthrough":
                assembler_cols.append(enc.column)
                
    # gather remaining numeric columns for assembly
    for field in train_data.schema.fields:
        if isinstance(field.dataType, (T.DoubleType, T.IntegerType, T.FloatType, T.LongType)):
            if (field.name not in assembler_cols and 
                field.name != "Severity" and 
                not field.name.endswith("_idx")):
                assembler_cols.append(field.name)

    final_feature_cols = list(dict.fromkeys([c for c in assembler_cols if c != "Severity"])) 

    # fit on train data
    pipeline = Pipeline(stages=stages)
    fitted_model = pipeline.fit(train_data)
    
    # transform on train and test data
    train_engineered = fitted_model.transform(train_data)
    test_engineered = fitted_model.transform(test_data)

    for enc in cat.column_encodings:
        if enc.action == "string_index_only":
            raw_col = f"{enc.column}_idx_raw"
            final_col = f"{enc.column}_idx"
            if raw_col in train_engineered.columns:
                train_engineered = train_engineered.withColumn(final_col, F.col(raw_col).cast("double")).drop(raw_col)
                test_engineered = test_engineered.withColumn(final_col, F.col(raw_col).cast("double")).drop(raw_col)
                
    #drop base columns
    cols_to_drop = set([d.column for d in cat.drop_after_encoding])
    for enc in cat.column_encodings:
        if enc.action in ["string_index_ohe", "string_index_only"]:
            cols_to_drop.add(enc.column)
        if enc.action == "string_index_ohe":
            cols_to_drop.add(f"{enc.column}_idx")
        if enc.action == "string_index_only":
            cols_to_drop.add(f"{enc.column}_idx_raw")
        if enc.action =="drop":
            cols_to_drop.add(enc.column)

    if num.bins:
        for b in num.bins:
            cols_to_drop.add(b.column)
            
    for exp in sem.semantic_expansions:
        cols_to_drop.add(exp.source_column)
        
    existing_drop_cols = [c for c in cols_to_drop if c in train_engineered.columns]
    if existing_drop_cols:
        train_engineered = train_engineered.drop(*existing_drop_cols)
        test_engineered = test_engineered.drop(*existing_drop_cols)
    
    # Regenerate final feature cols after dropping
    final_feature_cols = [col for col in final_feature_cols if col in train_engineered.columns]

    os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
    train_out_path = str(PROJECT_ROOT / "dataset" / "engineered_df_train.parquet")
    test_out_path = str(PROJECT_ROOT / "dataset" / "engineered_df_test.parquet")
    
    #save pipeline model for use in modelling stage
    model_out_path = str(PROJECT_ROOT / "models" / "fe_pipeline_model")

    train_engineered.write.mode("overwrite").parquet(train_out_path)
    test_engineered.write.mode("overwrite").parquet(test_out_path)
    fitted_model.write().overwrite().save(model_out_path)
    
    print(f"Feature engineering compiled and executed. Assembled {len(final_feature_cols)} features.")

    return {
        "feature_cols": final_feature_cols,
        "engineered_df_train_path": train_out_path,
        "engineered_df_test_path": test_out_path,
        "fe_pipeline_model_path": model_out_path
    }

def validate_feature_eng(state: AgentState, df) -> dict:
    validation_failures = state.get("validation_failures", [])
    if validation_failures is None:
        validation_failures = []

    results = {"passed": True, "checks": []}

    def check(name, condition, detail):
        status = "PASS" if condition else "FAIL"
        results["checks"].append({"check": name, "status": status, "detail": detail})
        if not condition:
            results["passed"] = False

    feature_cols = state.get("feature_cols", [])
    
    check(
        "no_target_leakage",
        "Severity" not in feature_cols,
        "Severity excluded from assembler_cols" if "Severity" not in feature_cols else "Severity present in assembler_cols"
    )

    #check for nulls
    for col in feature_cols:
        if col in df.columns:
            n_null = df.filter(F.col(col).isNull()).count()
            check(f"no_nulls_{col}", n_null == 0, f"{n_null} nulls found" if n_null > 0 else "Clean")
            
    # cehck time features are in expected range
    num_decisions = state.get("numeric_decisions", {})
    time_feats = num_decisions.get("time_features", {})
    if time_feats and "features_to_extract" in time_feats:
        for feat in time_feats["features_to_extract"]:
            if feat == "is_rush_hour" and feat in df.columns:
                out_of_range = df.filter((F.col(feat) < 0) | (F.col(feat) > 1)).count()
                check(f"range_{feat}", out_of_range == 0, f"{out_of_range} rows out of [0, 1]" if out_of_range else "All in [0, 1]")
    
    # check dropped columns are actually gone
    cat_decisions = state.get("categorical_decisions", {})
    drop_decisions = cat_decisions.get("drop_after_encoding", {})
    expected_drops = set([d["column"] for d in drop_decisions])

    #get expected cols to be dropped from semantic and categorical agent
    sem_decisions = state.get("semantic_decisions", {})
    for exp in sem_decisions.get("semantic_expansions", {}):
        expected_drops.add(exp['source_column'])
    
    num_decisions = state.get("numeric_decisions", {})
    #only checking for bin decision, interaction base cols do not need to be dropped. Timestamp cols handled by categorical agent
    for bin_dec in num_decisions.get("numeric_decisions",{}):
        expected_drops.add(bin_dec["column"])

    surviving_columns = [col for col in expected_drops if col in df.columns]
    check(
        "dropped_columns_removed",
        len(surviving_columns) == 0,
        "All columns meant to be dropped are dropped" if len(surviving_columns) == 0 else f"Failed to drop: {','.join(surviving_columns)}"
    )
    
    print("\n=== Feature Engineering Validation Report ===")
    for c in results["checks"]:
        symbol = "✓" if c["status"] == "PASS" else "x"
        print(f"[{symbol}] {c['check']}: {c['detail']}")
    print(f"\nOverall: {'PASSED' if results['passed'] else 'FAILED'}")

    failures = [
        f"{c['check']}: {c['detail']}"
        for c in results["checks"]
        if c["status"] == "FAIL"
    ]

    return {
        "feature_validation_report": results,
        "validation_passed": results["passed"],
        "validation_failures": failures,
    }

def profile_engineered_features(df, state: AgentState) -> dict:
    total_rows = df.count()
    target_col = "Severity_Binary" if "Severity_Binary" in df.columns else "Severity"
    
    profile = {
        "structural_overview": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "column_types": {f.name: str(f.dataType) for f in df.schema.fields}
        },
        "target_analysis": {},
        "numerical_analysis": {},
        "boolean_and_binary_analysis": {},
        "categorical_indices_analysis": {},
        "vector_analysis": {}
    }
    
    # 1. inspect target cols and class imbalance 
    if target_col in df.columns:
        class_dist = df.groupBy(target_col).count().orderBy(target_col).collect()
        profile["target_analysis"]["class_balance"] = {
            str(r[target_col]): {
                "count": r["count"],
                "pct": round(r["count"] / total_rows * 100, 1)
            } for r in class_dist
        }
        
        if "0" in profile["target_analysis"]["class_balance"] and "1" in profile["target_analysis"]["class_balance"]:
            n_maj = profile["target_analysis"]["class_balance"]["0"]["count"]
            n_min = profile["target_analysis"]["class_balance"]["1"]["count"]
            profile["target_analysis"]["class_weight_ratio"] = round(n_maj / max(n_min, 1), 2)

    # 2. Categorize Columns for Single-Pass Processing
    dtypes = dict(df.dtypes)
    num_cols = []
    bool_cols = []
    vec_cols = []
    
    for c, t in dtypes.items():
        if c == target_col: continue
        
        if "vector" in t.lower():
            vec_cols.append(c)
        elif c.endswith("_idx"):
            cardinality = df.select(F.approx_count_distinct(c)).collect()[0][0]
            profile["categorical_indices_analysis"][c] = {"approx_cardinality": cardinality}
        elif t == "boolean":
            bool_cols.append(c)
        elif t in ["int", "double", "float", "bigint"]:
            num_cols.append(c)

    # 3. boolean cols
    if bool_cols:
        bool_exprs = [F.sum(F.when(F.col(c) == True, 1).otherwise(0)).alias(c) for c in bool_cols]
        bool_res = df.select(bool_exprs).first().asDict()
        
        for c in bool_cols:
            trues = bool_res[c]
            profile["boolean_and_binary_analysis"][c] = {
                "true_count": trues,
                "true_ratio": round(trues / max(total_rows, 1), 4),
                "type": "native_boolean"
            }

    # 4. numerics & semantic binary flags   
    if num_cols:
        stats_exprs = []
        for c in num_cols:
            stats_exprs.extend([
                F.mean(c).alias(f"{c}_mean"),
                F.stddev(c).alias(f"{c}_std"),
                F.min(c).alias(f"{c}_min"),
                F.max(c).alias(f"{c}_max"),
                F.skewness(c).alias(f"{c}_skew")
            ])
            
        num_res = df.select(stats_exprs).first().asDict()
        
        for c in num_cols:
            mn = num_res[f"{c}_min"]
            mx = num_res[f"{c}_max"]
            mean_val = num_res[f"{c}_mean"]
            
            # If the integer column only contains 0 and 1 (like our semantic regex outputs), 
            # it is mathematically a binary flag, not a continuous variable.
            if mn == 0 and mx == 1:
                profile["boolean_and_binary_analysis"][c] = {
                    "true_ratio": round(mean_val, 4) if mean_val is not None else 0.0,
                    "type": "binary_integer"
                }
            else:
                std_val = num_res[f"{c}_std"]
                skew_val = num_res[f"{c}_skew"]
                
                profile["numerical_analysis"][c] = {
                    "min": round(mn, 4) if mn is not None else None,
                    "max": round(mx, 4) if mx is not None else None,
                    "mean": round(mean_val, 4) if mean_val is not None else None,
                    "stddev": round(std_val, 4) if std_val is not None else None,
                    "skewness": round(skew_val, 4) if skew_val is not None else None,
                    "contains_negatives": (mn < 0) if mn is not None else False
                }

    
    # 5. vector analysis (OHE Outputs & Final Features Vector)
    if vec_cols:
        # Grab exactly 1 row to inspect vector sizes (Zero computation cost)
        first_row = df.select(vec_cols).head()
        if first_row:
            for c in vec_cols:
                vec = first_row[c]
                if vec is not None:
                    profile["vector_analysis"][c] = {
                        "vector_type": "SparseVector" if hasattr(vec, 'indices') else "DenseVector",
                        "vector_size": len(vec)
                    }

    return {"feature_stats_report": profile}

MAX_VALIDATION_RETRIES = 2

def validate_feature_eng_node(state: AgentState):
    df = spark.read.parquet(state["engineered_df_train_path"])

    val_report = validate_feature_eng(state, df)
    updated_state = {**state, **val_report}

    if updated_state["validation_passed"]:
        stats_report = profile_engineered_features(df, updated_state)
        updated_state.update(stats_report)
    else:
        updated_state["validation_iterations"] = state.get("validation_iterations", 0) + 1

    return updated_state
    fe_decision = (state.get("feature_auto_decision") or {}).get("decision")
    if fe_decision == "reject":
        if state.get("validation_iterations", 0) <= MAX_VALIDATION_RETRIES:
            return "retry"
    return "end"

def feature_eng_accept_reject_node(state: AgentState) -> AgentState:
    validation_passed = bool(state.get("validation_passed", False))
    feature_cols = state.get("feature_cols", []) or []
    feature_stats = state.get("feature_stats_report", {}) or {}
    sem_decisions = (state.get("semantic_decisions", {}) or {}).get("semantic_expansions", []) or []

    quality_issues = []
    warnings = []

    if not validation_passed:
        quality_issues.append("Feature validation checks failed")

    if not feature_cols:
        quality_issues.append("No engineered feature columns were produced")

    bool_profile = feature_stats.get("boolean_and_binary_analysis", {}) if isinstance(feature_stats, dict) else {}
    suspicious_semantic_flags = []

    for exp in sem_decisions:
        target_col = exp.get("target_column")
        regex = str(exp.get("regex_pattern", "")).strip()

        if not target_col:
            quality_issues.append("Semantic expansion has missing target_column")
            continue

        if target_col not in bool_profile:
            suspicious_semantic_flags.append({
                "target_column": target_col,
                "reason": "engineered semantic flag not found in final profile"
            })
            continue

        true_ratio = bool_profile[target_col].get("true_ratio")
        if isinstance(true_ratio, (int, float)):
            if true_ratio < 0.001:
                suspicious_semantic_flags.append({
                    "target_column": target_col,
                    "reason": f"true_ratio too low ({true_ratio:.4f}) -> likely over-specific regex"
                })
            elif true_ratio > 0.995:
                suspicious_semantic_flags.append({
                    "target_column": target_col,
                    "reason": f"true_ratio too high ({true_ratio:.4f}) -> likely over-broad regex"
                })

        if regex and len(regex) < 3:
            warnings.append(f"Semantic regex for {target_col} looks too short: '{regex}'")

    if sem_decisions and len(suspicious_semantic_flags) > max(3, int(len(sem_decisions) * 0.4)):
        quality_issues.append(
            "Too many semantic boolean mappings look suspicious; feature engineering likely needs revision"
        )

    accepted = len(quality_issues) == 0
    decision = "accept" if accepted else "reject"
    status = "approved" if accepted else "needs_more_feature_engineering"
    cols_to_prune = []
    
    #extract sus columns so that can be dropped in subsequent node, this is to avoid noise in the final dataset
    if accepted and suspicious_semantic_flags:
        cols_to_prune = [flag['target_column'] for flag in suspicious_semantic_flags]
        state["cols_to_prune"] = cols_to_prune

    if not accepted:
        state["validation_iterations"] = state.get("validation_iterations", 0) + 1

    summary_reason = (
        "Engineered features accepted for modeling."
        if accepted
        else "Engineered features rejected; additional feature engineering is recommended."
    )

    fe_auto_decision = {
        "decision": decision,
        "status": status,
        "validation_passed": validation_passed,
        "semantic_flags_reviewed": len(sem_decisions),
        "suspicious_semantic_flags": suspicious_semantic_flags,
        "warning_count": len(warnings),
        "quality_issues": quality_issues,
        "reason": summary_reason,
    }

    state["feature_auto_decision"] = fe_auto_decision
    state["feature_validation_status"] = status
    state["feature_validation_message"] = summary_reason
    state.setdefault("execution_log", []).append({"node": "feature_eng_gate", **fe_auto_decision})

    print("=" * 55)
    print(">> NODE: Feature engineering accept/reject gate")
    print("=" * 55)
    print(json.dumps(fe_auto_decision, indent=2))

    return state

def prune_features_node(state: AgentState) -> AgentState:
    """
    this takes in the suspicious columns identified in the feature gate and drops them
    updates feature_cols, feature_stats_report and feature_validation_report to avoid stale state
    """
    cols_to_prune = state.get("cols_to_prune") or []
    if not cols_to_prune:
        print("No columns to prune. Proceeding...")
        return state
    
    print(f"\n" + "="*60)
    print(f">> NODE: Auto-Pruning {len(cols_to_prune)} sparse features")
    print(f"="*60)
    print(f"Dropping physical columns: {cols_to_prune}")

    #update engineered dataframe and rewrite to parquet
    train_df = spark.read.parquet(state["engineered_df_train_path"])
    existing_cols_train = [c for c in cols_to_prune if c in train_df.columns]
    if existing_cols_train:
        train_df = train_df.drop(*existing_cols_train)
        new_train_path = str(PROJECT_ROOT / "dataset" / "engineered_df_train_pruned.parquet")
        train_df.write.mode("overwrite").parquet(new_train_path)
        state['engineered_df_train_path'] = new_train_path
        

    test_df = spark.read.parquet(state["engineered_df_test_path"])
    existing_cols_test = [c for c in cols_to_prune if c in test_df.columns]
    if existing_cols_test:
        test_df = test_df.drop(*existing_cols_test)
        new_test_path = str(PROJECT_ROOT / "dataset" / "engineered_df_test_pruned.parquet")
        test_df.write.mode("overwrite").parquet(new_test_path)
        state['engineered_df_test_path'] = new_test_path
        
    print("Feature engineered train and test parquest data succesfully pruned and saved")
    
    #update feature cols
    state['feature_cols'] = [c for c in state.get("feature_cols", []) if c not in cols_to_prune]

    #update feature stats report
    stats = state.get("feature_stats_report", {})
    if stats:
        #update boolean section
        bool_profile = stats.get("boolean_and_binary_analysis", {})
        for col in cols_to_prune:
            bool_profile.pop(col, None)

        #update overview section
        structure = stats.get("structural_overview", {})
        col_types = structure.get("column_types", {})
        for col in cols_to_prune:
            col_types.pop(col, None)
        
        if "total_columns" in structure:
            structure['total_columns'] = len(col_types)
    
    #update feature_validation_report
    val_report = state.get("feature_validation_report")
    if "checks" in val_report:
        val_report["checks"] = [
            check for check in val_report["checks"]
            if not any(pruned_col in check["check"] for pruned_col in cols_to_prune)
        ]
    print("AgentState succesfully updated after pruning suspicious columns")
    print("="*60 + "\n")
    return state

def route_after_validation(state: AgentState):
    if not state.get("validation_passed", False):
        if state.get("validation_iterations", 0) <= MAX_VALIDATION_RETRIES:
            return "retry"
        return "end"
    return "feature_eng_gate"

def route_after_feature_gate(state: AgentState):
    fe_decision = (state.get("feature_auto_decision") or {}).get("decision")
    if fe_decision == "reject":
        if state.get("validation_iterations", 0) <= MAX_VALIDATION_RETRIES:
            return "retry"
    return "end"

