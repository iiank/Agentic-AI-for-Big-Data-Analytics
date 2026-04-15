"""
test_evaluation.py
Tests agentic decision-making in isolation — no Spark, no model training.
Usage: python3 test_evaluation.py

What this covers (no Spark needed):
  1. evaluation_node + routing: weak models (high FN, low AUC-PR)  → expect retrain
  2. evaluation_node + routing: strong AUC-PR but low recall@0.5   → expect tune_threshold
  3. evaluation_node + routing: good overall results                → expect accept
  4. evaluation_node + routing: max iterations hit                  → expect accept regardless
  5. Retry loop: retrain → model_selection_node with previous_attempt, no model repeats
  6. Feature importances print block: LR (None) and RF (from disk)

What this does NOT cover (needs Spark/LangGraph runtime):
  - evaluation_review_node  — uses interrupt(), requires full graph
  - threshold_tuning_node   — needs Spark + saved model
  - run_full_evaluation     — needs Spark + saved model
"""

import json
from pathlib import Path
from langgraph.graph import END

from Modelling_Agent import (
    evaluation_node,
    route_after_evaluation,
    model_selection_node,
    AgentState,
)

SAVED_RF_FI = Path(__file__).parent / "saved_models" / "RandomForestClassifier" / "feature_importances.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cm(tp, fn, fp, tn):
    return {"true_positives": tp, "false_negatives": fn, "false_positives": fp, "true_negatives": tn}

def _results(model_name, auc_pr, auc_roc, recall, precision, tp, fn, fp, tn):
    return {
        "model_name":       model_name,
        "auc_pr":           auc_pr,
        "auc_roc":          auc_roc,
        "recall_class1":    recall,
        "precision_class1": precision,
        "confusion_matrix": _cm(tp, fn, fp, tn),
        "best_params":      {},
        "feature_importances": None,
        "missing_features": [],
    }

def _base_state(primary, secondary, iteration=0) -> AgentState:
    return {
        "train_df": None,
        "test_df": None,
        "iteration":             iteration,
        "feature_cols":          ["is_rush_hour", "Start_Time_Hour", "Visibility(mi)_bin"],
        "post_cleaning_profile": {
            "num_rows": 7183644,
            "class_distribution": {
                0: {"count": 5774831, "pct": 80.39},
                1: {"count": 1408813, "pct": 19.61},
            },
            "class_weight_ratio": 4.1,
        },
        "model_selection":     {"primary_model": primary["model_name"], "secondary_model": secondary["model_name"]},
        "primary_results":     primary,
        "secondary_results":   secondary,
        "evaluation_decision": {},
        "tuning_results":      {},
        "modelling_log":       [],
    }


def run_eval(label, state):
    print("\n" + "=" * 60)
    print(f"TEST: {label}")
    print("=" * 60)
    result      = evaluation_node(state)
    # merge returned partial dict into state for routing
    merged      = {**state, **result}
    decision    = merged["evaluation_decision"]
    next_node   = route_after_evaluation(merged)
    winner      = decision.get("winner")
    print(f"  winner      : {winner}")
    print(f"  next_action : {decision.get('next_action')}")
    print(f"  routes to   : {next_node}")
    if decision.get("retrain_guidance"):
        print(f"  guidance    : {decision.get('retrain_guidance')}")
    if decision.get("tune_guidance"):
        print(f"  tune note   : {decision.get('tune_guidance')}")
    return merged


# ── Tests 1–4: evaluation_node + routing ─────────────────────────────────────

# Test 1: Both models weak — low AUC-PR, high false negatives → expect retrain
run_eval(
    "1. Weak models — low AUC-PR, high FN → expect retrain",
    _base_state(
        _results("LogisticRegression",     auc_pr=0.31, auc_roc=0.67, recall=0.28, precision=0.45,
                 tp=82000,  fn=192000, fp=98000,  tn=1032000),
        _results("RandomForestClassifier", auc_pr=0.38, auc_roc=0.72, recall=0.33, precision=0.48,
                 tp=96000,  fn=178000, fp=104000, tn=1026000),
        iteration=0,
    )
)

# Test 2: Strong AUC-PR but low recall at 0.5 → expect tune_threshold
run_eval(
    "2. Strong AUC-PR, low recall@0.5 → expect tune_threshold",
    _base_state(
        _results("LogisticRegression",     auc_pr=0.55, auc_roc=0.80, recall=0.36, precision=0.62,
                 tp=105000, fn=169000, fp=64000,  tn=1066000),
        _results("RandomForestClassifier", auc_pr=0.68, auc_roc=0.85, recall=0.41, precision=0.65,
                 tp=120000, fn=154000, fp=65000,  tn=1065000),
        iteration=0,
    )
)

# Test 3: Good results → expect accept
run_eval(
    "3. Good results — strong AUC-PR and recall → expect accept",
    _base_state(
        _results("GBTClassifier",          auc_pr=0.76, auc_roc=0.88, recall=0.68, precision=0.61,
                 tp=198000, fn=76000,  fp=127000, tn=1003000),
        _results("RandomForestClassifier", auc_pr=0.71, auc_roc=0.85, recall=0.62, precision=0.59,
                 tp=181000, fn=93000,  fp=126000, tn=1004000),
        iteration=0,
    )
)

# Test 4: Max iterations hit → expect accept regardless of weak metrics
run_eval(
    "4. Weak models but max iterations hit → expect accept",
    _base_state(
        _results("LogisticRegression",     auc_pr=0.31, auc_roc=0.67, recall=0.28, precision=0.45,
                 tp=82000,  fn=192000, fp=98000,  tn=1032000),
        _results("RandomForestClassifier", auc_pr=0.38, auc_roc=0.72, recall=0.33, precision=0.48,
                 tp=96000,  fn=178000, fp=104000, tn=1026000),
        iteration=2,
    )
)


# ── Test 5: Retry loop ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 5: Retry loop — model_selection_node receives previous_attempt, no repeats")
print("=" * 60)

_retrain_state = _base_state(
    _results("LogisticRegression",     auc_pr=0.31, auc_roc=0.67, recall=0.28, precision=0.45,
             tp=82000, fn=192000, fp=98000, tn=1032000),
    _results("RandomForestClassifier", auc_pr=0.38, auc_roc=0.72, recall=0.33, precision=0.48,
             tp=96000, fn=178000, fp=104000, tn=1026000),
    iteration=0,
)

_after_eval        = evaluation_node(_retrain_state)
_merged_after_eval = {**_retrain_state, **_after_eval, "iteration": 1}

print(f"  next_action      : {_merged_after_eval['evaluation_decision'].get('next_action')}")
print(f"  retrain_guidance : {_merged_after_eval['evaluation_decision'].get('retrain_guidance')}")
print("\n  Passing to model_selection_node with previous_attempt context...")

_after_selection = model_selection_node(_merged_after_eval)
_new             = _after_selection["model_selection"]
print(f"  new primary   : {_new.get('primary_model')}")
print(f"  new secondary : {_new.get('secondary_model')}")

_avoided = {
    _retrain_state["primary_results"]["model_name"],
    _retrain_state["secondary_results"]["model_name"],
}
_chosen  = {_new.get("primary_model"), _new.get("secondary_model")}
_overlap = _avoided & _chosen
if _overlap:
    print(f"  WARNING: repeated previous models: {_overlap}")
else:
    print(f"  OK: no repeated models from previous attempt")


# ── Test 6: Feature importances print block ───────────────────────────────────

print("\n" + "=" * 60)
print("TEST 6: Feature importances print block — LR (None) and RF (from disk)")
print("=" * 60)

_lr_results = _results("LogisticRegression", 0.31, 0.67, 0.28, 0.45, 82000, 192000, 98000, 1032000)
_lr_results["feature_importances"] = None

_rf_fi = json.loads(SAVED_RF_FI.read_text()) if SAVED_RF_FI.exists() else None
_rf_results = _results("RandomForestClassifier", 0.38, 0.72, 0.33, 0.48, 96000, 178000, 104000, 1026000)
_rf_results["feature_importances"] = _rf_fi

for role, res in [("primary", _lr_results), ("secondary", _rf_results)]:
    if res.get("feature_importances"):
        print(f"\n  Top-10 importances ({res['model_name']}):")
        for feat, imp in list(res["feature_importances"].items())[:10]:
            print(f"    {feat:<45} {imp:.4f}")
    else:
        print(f"  {res['model_name']}: feature_importances=None — skipped (expected for LR)")


print("\n[test_evaluation.py] All 6 tests done.")
