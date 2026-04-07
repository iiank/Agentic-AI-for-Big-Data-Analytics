# modelling_agent.py
# Orchestration layer for the modelling agent.
# Defines AgentState, LangGraph nodes, LLM calls, and graph wiring.
# Imports PySpark execution from modelling_pyspark_skills.py.
# The LLM decisions drive what the skills file executes.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from pyspark.sql import SparkSession

from modelling import MODEL_REGISTRY, run_training


# ── Spark Session ─────────────────────────────────────────────────────────────

if "spark" in locals():
    spark.stop()
    print("Existing Spark session stopped.")

spark = SparkSession.builder \
    .appName("BT4221_Modelling") \
    .master("local[*]") \
    .config("spark.driver.memory", "12g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.shuffle.partitions", "48") \
    .config("spark.default.parallelism", "16") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark running:", spark.version)


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT           = Path.cwd()
ENGINEERED_PARQUET_DIR = PROJECT_ROOT / "engineered_parquet"   # output from Phase 3
SKILL_PROMPT_PATH      = PROJECT_ROOT / "modelling_agent_prompt.md"

if not ENGINEERED_PARQUET_DIR.exists():
    raise FileNotFoundError(
        f"Engineered parquet not found at: {ENGINEERED_PARQUET_DIR}\n"
        "Update ENGINEERED_PARQUET_DIR to the output path from Phase 3."
    )

engineered_df = spark.read.parquet(str(ENGINEERED_PARQUET_DIR))
print(f"Loaded engineered dataset: {engineered_df.count():,} rows")

SKILL_PROMPT = SKILL_PROMPT_PATH.read_text()


# ── AgentState ────────────────────────────────────────────────────────────────
# Extends the shared pipeline state from Phases 1-3.
# Phases 1-3 must provide: cleaning_summary, post_cleaning_profile, feature_cols.

class AgentState(TypedDict):
    # Inherited from Phases 1-3
    cleaning_summary:       str
    post_cleaning_profile:  Dict[str, Any]
    feature_cols:           List[str]

    # Modelling agent fields
    auc_threshold:          float           # e.g. 0.80
    max_retries:            int             # graph will not loop past this
    retry_count:            int             # incremented after each evaluation
    model_selection:        Dict[str, Any]  # LLM Call 1 output
    training_results:       Dict[str, Any]  # AUC, best params, feature importances
    evaluation_decision:    Dict[str, Any]  # LLM Call 2 output
    modelling_log:          List[str]       # append-only audit trail


# ── LLM Client ────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(user_payload: Dict) -> Dict:
    """
    Send a context payload to GPT-4o with the modelling skill prompt.
    Returns parsed JSON. Temperature=0 for deterministic decisions.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SKILL_PROMPT},
            {"role": "user",   "content": json.dumps(user_payload, indent=2)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# ── Context Builders ──────────────────────────────────────────────────────────
# These construct the small structured summaries the LLM reasons from.
# Designed to surface decision-relevant facts, not a generic data dump.

def build_model_selection_context(state: AgentState) -> Dict:
    profile    = state["post_cleaning_profile"]
    class_dist = profile.get("class_distribution", {})
    majority_pct = max(v["pct"] for v in class_dist.values()) if class_dist else None

    return {
        # Task definition
        "target_col":    "Severity_Binary",
        "target_type":   "binary",
        "auc_threshold": state["auc_threshold"],

        # Dataset scale
        "num_rows":     profile.get("num_rows"),
        "num_features": len(state["feature_cols"]),
        "feature_cols": state["feature_cols"],

        # Class imbalance -- most decision-relevant statistic
        "class_weight_ratio": profile.get("class_weight_ratio"),
        "majority_class_pct": majority_pct,
        "minority_class_pct": min(
            v["pct"] for v in class_dist.values()
        ) if class_dist else None,
        "naive_classifier_note": (
            f"Always predicting the majority class achieves {majority_pct:.2f}% accuracy. "
            f"AUC must substantially exceed 0.5 to be meaningful."
            if majority_pct else None
        ),

        # Execution environment (informs grid size constraint)
        "spark_mode": "local[*] on MacBook Air M2 -- constrain grid to 4-6 combinations",

        # Registry exposed to LLM for selection and exclusion reasoning
        "available_models": {
            name: {
                "task":           info["task"],
                "tunable_params": list(info["tunable"].keys()),
            }
            for name, info in MODEL_REGISTRY.items()
        },

        # Retry context (populated on subsequent loops)
        "retry_count":    state["retry_count"],
        "previous_model": state["model_selection"].get("selected_model")
                          if state.get("model_selection") else None,
        "previous_auc":   state["training_results"].get("auc")
                          if state.get("training_results") else None,
    }


def build_evaluation_context(state: AgentState) -> Dict:
    return {
        "auc_achieved":   state["training_results"]["auc"],
        "auc_threshold":  state["auc_threshold"],
        "best_params":    state["training_results"].get("best_params", {}),
        "selected_model": state["model_selection"]["selected_model"],
        "retry_count":    state["retry_count"],
        "max_retries":    state["max_retries"],
        "retry_model":    state["model_selection"].get("retry_model"),
    }


# ── LangGraph Nodes ───────────────────────────────────────────────────────────

def model_selection_node(state: AgentState) -> AgentState:
    print("\n[Modelling Agent] Node: model_selection")

    context  = build_model_selection_context(state)
    decision = call_llm(context)
    state["model_selection"] = decision

    state["modelling_log"] = state.get("modelling_log", []) + [
        f"[model_selection] retry={state['retry_count']} | "
        f"selected={decision['selected_model']} | "
        f"excluded={list(decision.get('excluded_models', {}).keys())}"
    ]

    print(f"  Selected : {decision['selected_model']}")
    print(f"  Why      : {decision['justification']}")
    print(f"  Grid     : {decision.get('param_grid')}")
    print(f"  Excluded : {list(decision.get('excluded_models', {}).keys())}")
    return state


def human_review_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop gate before expensive training.
    Uses LangGraph interrupt() to pause execution and return control to the caller.
    Caller resumes with Command(resume={"approved": True}) or
    Command(resume={"approved": False, "override": "ModelName"}).
    """
    decision = state["model_selection"]

    # Pause here — execution freezes until caller resumes with Command(resume=...)
    human_input = interrupt({
        "model":           decision["selected_model"],
        "justification":   decision["justification"],
        "param_grid":      decision.get("param_grid"),
        "param_grid_note": decision.get("param_grid_rationale"),
        "retry_model":     decision.get("retry_model"),
        "excluded_models": decision.get("excluded_models", {}),
    })

    approved = human_input.get("approved", True)
    override = human_input.get("override", "").strip()

    if not approved and override:
        if override in MODEL_REGISTRY:
            state["model_selection"]["selected_model"] = override
            state["model_selection"]["justification"]  = f"[Human override] {override} selected by user."
        else:
            print(f"  '{override}' not in registry. Proceeding with agent selection.")

    state["modelling_log"] = state["modelling_log"] + [
        f"[human_review] approved={approved} | "
        f"final_model={state['model_selection']['selected_model']}"
    ]
    return state


def training_node(state: AgentState) -> AgentState:
    """
    Calls run_training() from modelling_pyspark_skills.py.
    The LLM decision in model_selection drives what gets executed here.
    """
    decision   = state["model_selection"]
    model_name = decision["selected_model"]
    param_grid = decision.get("param_grid", {})

    print(f"\n[Modelling Agent] Node: training | model={model_name}")

    results = run_training(
        df=engineered_df,
        model_name=model_name,
        param_grid_spec=param_grid,
        feature_cols=state["feature_cols"],
    )
    state["training_results"] = results

    state["modelling_log"] = state["modelling_log"] + [
        f"[training] model={model_name} | "
        f"auc={results['auc']} | "
        f"best_params={results['best_params']}"
    ]

    print(f"  AUC         : {results['auc']}")
    print(f"  Best params : {results['best_params']}")
    if results.get("feature_importances"):
        top5 = list(results["feature_importances"].items())[:5]
        print(f"  Top-5 feats : {top5}")
    return state


def evaluation_node(state: AgentState) -> AgentState:
    """
    LLM Call 2. Receives AUC result, decides accept / retry / escalate.
    Also produces a plain-English narrative for the project report.
    """
    print("\n[Modelling Agent] Node: evaluation")

    context  = build_evaluation_context(state)
    decision = call_llm(context)
    state["evaluation_decision"] = decision
    state["retry_count"]         = state["retry_count"] + 1

    state["modelling_log"] = state["modelling_log"] + [
        f"[evaluation] decision={decision['decision']} | "
        f"auc={decision['auc_achieved']} | "
        f"threshold={decision['auc_threshold']}"
    ]

    print(f"  Decision  : {decision['decision'].upper()}")
    print(f"  AUC       : {decision['auc_achieved']} vs threshold {decision['auc_threshold']}")
    print(f"  Narrative : {decision['narrative']}")

    if decision["decision"] == "retry":
        state["model_selection"]["selected_model"] = decision.get("retry_model")
        print(f"  Next model: {decision.get('retry_model')}")

    if decision["decision"] == "escalate":
        print("\n  [!] Retries exhausted. Escalating to human.")
        print(f"      Best AUC seen: {decision['auc_achieved']}")

    return state


# ── Conditional Edge ──────────────────────────────────────────────────────────

def route_after_evaluation(
    state: AgentState,
) -> Literal["model_selection_node", "__end__"]:
    decision = state["evaluation_decision"]["decision"]
    if decision == "accept":
        return "__end__"
    if decision == "retry" and state["retry_count"] < state["max_retries"]:
        return "model_selection_node"
    return "__end__"


# ── Build and Compile Graph ───────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("model_selection_node", model_selection_node)
workflow.add_node("human_review_node",    human_review_node)
workflow.add_node("training_node",        training_node)
workflow.add_node("evaluation_node",      evaluation_node)

workflow.set_entry_point("model_selection_node")
workflow.add_edge("model_selection_node", "human_review_node")
workflow.add_edge("human_review_node",    "training_node")
workflow.add_edge("training_node",        "evaluation_node")
workflow.add_conditional_edges(
    "evaluation_node",
    route_after_evaluation,
    {
        "model_selection_node": "model_selection_node",
        "__end__":               END,
    },
)

checkpointer    = MemorySaver()
modelling_agent = workflow.compile(checkpointer=checkpointer)
print("Modelling agent graph compiled.")


# ── Run ───────────────────────────────────────────────────────────────────────
# Replace hardcoded values with final_state from Phase 3 if running
# all phases sequentially in the same notebook session.

initial_state: AgentState = {
    # From Phase 3 final_state
    "cleaning_summary": (
        "Manually cleaned data: handled nulls. "
        "Mapped Severity to Severity_Binary (Classes 1 & 2 -> 0; Classes 3 & 4 -> 1)."
    ),
    "post_cleaning_profile": {
        "num_rows": 5270673,
        "class_distribution": {
            0: {"count": 4531863, "pct": 85.98},
            1: {"count": 738810,  "pct": 14.02},
        },
        "class_weight_ratio": 6.13,
    },
    "feature_cols": [
        "is_rush_hour", "Weather_Condition_idx", "Wind_Direction_ohe",
        "State_ohe", "Sunrise_Sunset_ohe", "Start_Time_Hour",
        "Start_Time_is_Weekend", "Visibility(mi)_bin", "Temperature(F)_bin",
        "Wind_Speed(mph)_bin", "Distance(mi)", "Duration_Minutes",
        "Distance(mi)_ratio_Duration_Minutes",
    ],

    # Modelling agent config
    "auc_threshold":       0.80,
    "max_retries":         3,
    "retry_count":         0,
    "model_selection":     {},
    "training_results":    {},
    "evaluation_decision": {},
    "modelling_log":       [],
}

print("\nStarting Modelling Agent...\n")
config = {"configurable": {"thread_id": "modelling_run_1"}}

modelling_agent.invoke(initial_state, config=config)

# Resume loop: handles one human_review interrupt per training attempt
while modelling_agent.get_state(config).next:
    snapshot     = modelling_agent.get_state(config)
    interrupt_val = snapshot.tasks[0].interrupts[0].value

    print("\n" + "=" * 60)
    print("HUMAN REVIEW -- Model Selection")
    print("=" * 60)
    print(f"  Model      : {interrupt_val['model']}")
    print(f"  Why        : {interrupt_val['justification']}")
    print(f"  Param grid : {json.dumps(interrupt_val.get('param_grid'), indent=4)}")
    print(f"  Grid note  : {interrupt_val.get('param_grid_note')}")
    print(f"  If fails   : retry with {interrupt_val.get('retry_model')}")
    print("\n  Excluded models:")
    for model, reason in interrupt_val.get("excluded_models", {}).items():
        print(f"    {model}: {reason}")
    print("=" * 60)

    approval = input("\nApprove? (y to proceed, n to override model): ").strip().lower()

    if approval == "y":
        resume = {"approved": True}
    else:
        override = input("Enter model name to use instead: ").strip()
        resume   = {"approved": False, "override": override}

    modelling_agent.invoke(Command(resume=resume), config=config)

final_state = modelling_agent.get_state(config).values


# ── Print Results ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MODELLING AGENT COMPLETE")
print("=" * 60)
print(f"Final model  : {final_state['training_results'].get('model_name')}")
print(f"Final AUC    : {final_state['training_results'].get('auc')}")
print(f"Best params  : {final_state['training_results'].get('best_params')}")
print(f"Decision     : {final_state['evaluation_decision'].get('decision', '').upper()}")
print(f"\nNarrative (for report):\n{final_state['evaluation_decision'].get('narrative')}")

print("\nFull audit log:")
for entry in final_state["modelling_log"]:
    print(f"  {entry}")

if final_state["training_results"].get("feature_importances"):
    print("\nTop-10 feature importances:")
    for feat, imp in list(
        final_state["training_results"]["feature_importances"].items()
    )[:10]:
        print(f"  {feat:<45} {imp:.4f}")