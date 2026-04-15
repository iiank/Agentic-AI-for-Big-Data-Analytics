"""
test_evaluation.py
Tests post-training logic in isolation — no Spark, no model training.
Usage: python3 test_evaluation.py

Covers:
  1. evaluation_node: both below threshold           → expect retrain
  2. evaluation_node: winner above threshold         → expect accept
  3. evaluation_node: max iterations hit             → expect accept
  4. Retry loop: evaluation retrain → model_selection_node with previous_attempt
  5. Print results block: None feature importances (LR)
  6. Print results block: non-None feature importances (mocked RF)
"""

import json
from pathlib import Path
from Trial_Modelling_Agent import (
    evaluation_node, route_after_evaluation,
    model_selection_node, AgentState
)

SAVED_LR = Path(__file__).parent / "saved_models" / "LogisticRegression" / "results.json"
SAVED_RF = Path(__file__).parent / "saved_models" / "RandomForestClassifier" / "results.json"
SAVED_RF_FI = Path(__file__).parent / "saved_models" / "RandomForestClassifier" / "feature_importances.json"

def _base_state(primary, secondary, iteration) -> AgentState:
    return {
        "train_df": None,
        "test_df": None,
        "iteration":             iteration,
        "feature_cols":          ["is_rush_hour", "Start_Time_Hour", "Visibility(mi)_bin"],
        "post_cleaning_profile": {
            "num_rows": 5270673,
            "class_distribution": {
                0: {"count": 4531863, "pct": 85.98},
                1: {"count": 738810,  "pct": 14.02},
            },
            "class_weight_ratio": 6.13,
        },
        "model_selection":       {},
        "modelling_log":         [],
        "primary_results":   {**primary,   "best_params": {}, "feature_importances": None, "missing_features": []},
        "secondary_results": {**secondary, "best_params": {}, "feature_importances": None, "missing_features": []},
        "evaluation_decision":   {},
    }


def run_eval(label, state):
    print("\n" + "=" * 60)
    print(f"TEST: {label}")
    print("=" * 60)
    result    = evaluation_node(state)
    decision  = result["evaluation_decision"]
    next_node = route_after_evaluation(result)
    print(f"  winner      : {decision.get('winner')} (AUC {decision.get('winner_auc')})")
    print(f"  next_action : {decision.get('next_action')}")
    print(f"  routes to   : {next_node}")
    if decision.get("retrain_guidance"):
        print(f"  guidance    : {decision.get('retrain_guidance')}")
    return result


# ── Tests 1–3: evaluation_node routing ───────────────────────────────────────  

run_eval(
    "1. Both below threshold (iter 0) → expect retrain  [real saved AUCs]",
    _base_state(
        {"model_name": "LogisticRegression",     "auc": 0.669},
        {"model_name": "RandomForestClassifier", "auc": 0.7811},
        iteration=0,
    )
)

run_eval(
    "2. Winner above threshold → expect accept",
    _base_state(
        {"model_name": "GBTClassifier",      "auc": 0.821},
        {"model_name": "LogisticRegression", "auc": 0.669},
        iteration=0,
    )
)

run_eval(
    "3. Both below threshold but max iterations hit → expect accept",
    _base_state(
        {"model_name": "LogisticRegression",     "auc": 0.669},
        {"model_name": "RandomForestClassifier", "auc": 0.793},
        iteration=2,
    )
)


# ── Test 4: Retry loop — evaluation retrain feeds into model_selection ────────

print("\n" + "=" * 60)
print("TEST 4: Retry loop — model_selection_node receives previous_attempt")
print("=" * 60)

_retrain_state = _base_state(
    {"model_name": "LogisticRegression",     "auc": 0.669},
    {"model_name": "RandomForestClassifier", "auc": 0.793},
    iteration=0,
)
_after_eval = evaluation_node(_retrain_state)
_after_eval["iteration"] = 1  # simulate post-increment

print(f"  evaluation next_action : {_after_eval['evaluation_decision'].get('next_action')}")
print(f"  retrain_guidance       : {_after_eval['evaluation_decision'].get('retrain_guidance')}")
print("\n  Passing to model_selection_node with previous_attempt context...")

_after_selection = model_selection_node(_after_eval)
_new = _after_selection["model_selection"]
print(f"  new primary   : {_new.get('primary_model')}")
print(f"  new secondary : {_new.get('secondary_model')}")

_avoided = {_retrain_state["primary_results"]["model_name"], _retrain_state["secondary_results"]["model_name"]}
_chosen  = {_new.get("primary_model"), _new.get("secondary_model")}
_overlap = _avoided & _chosen
if _overlap:
    print(f"  WARNING: model_selection_node repeated previous models: {_overlap}")
else:
    print(f"  OK: no repeated models from previous attempt")


# ── Test 5: Print block — None feature importances (matches real LR save) ─────

print("\n" + "=" * 60)
print("TEST 5: Print results block — None feature importances (LR)")
print("=" * 60)

_lr_results = json.loads(SAVED_LR.read_text()) if SAVED_LR.exists() else {
    "model_name": "LogisticRegression", "auc": 0.669,
    "best_params": {"regParam": 0.01}, "missing_features": []
}
_lr_results["feature_importances"] = None

_fake_final = {
    "evaluation_decision": {"winner": "RandomForestClassifier", "winner_auc": 0.793,
                            "runner_up": "LogisticRegression",  "runner_up_auc": 0.669,
                            "narrative": "Test narrative.", "next_action": "retrain"},
    "modelling_log": ["[model_selection] primary=LR | secondary=RF",
                      "[training_primary] auc=0.669", "[training_secondary] auc=0.793"],
    "primary_results":   _lr_results,
    "secondary_results": {"model_name": "RandomForestClassifier", "auc": 0.793,
                          "best_params": {}, "feature_importances": None, "missing_features": []},
}

for role in ("primary", "secondary"):
    res = _fake_final[f"{role}_results"]
    if res.get("feature_importances"):
        print(f"  Top-10 importances ({res['model_name']}):")
        for feat, imp in list(res["feature_importances"].items())[:10]:
            print(f"    {feat:<45} {imp:.4f}")
    else:
        print(f"  {res['model_name']}: feature_importances=None — skipped (expected for LR)")


# ── Test 6: Print block — real RF feature importances from disk ───────────────

print("\n" + "=" * 60)
print("TEST 6: Print results block — real RF feature importances from disk")
print("=" * 60)

_rf_results = json.loads(SAVED_RF.read_text())
_rf_results["feature_importances"] = json.loads(SAVED_RF_FI.read_text()) if SAVED_RF_FI.exists() else None
_fake_final["secondary_results"] = _rf_results

for role in ("primary", "secondary"):
    res = _fake_final[f"{role}_results"]
    if res.get("feature_importances"):
        print(f"  Top-10 importances ({res['model_name']}):")
        for feat, imp in list(res["feature_importances"].items())[:10]:
            print(f"    {feat:<45} {imp:.4f}")
    else:
        print(f"  {res['model_name']}: feature_importances=None — skipped")


print("\n[test_evaluation.py] All 6 tests done.")
