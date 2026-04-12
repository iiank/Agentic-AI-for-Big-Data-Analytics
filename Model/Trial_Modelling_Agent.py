# Trial_Modelling_Agent.py
# Orchestration layer for the modelling agent.
# Defines AgentState, LangGraph nodes, LLM calls, and graph wiring.
# Imports PySpark execution from Modelling_Skills.py.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from Modelling_Skills import MODEL_REGISTRY, run_training


# ── LLM Client ────────────────────────────────────────────────────────────────

client = OpenAI(api_key="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")

SKILL_PROMPT = (Path(__file__).parent / "Modelling_Prompt.md").read_text()


# ── AgentState ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input — pass in from FE phase
    feature_cols:          List[str]
    post_cleaning_profile: Dict[str, Any]
    iteration:             int              # retry counter

    # Agent outputs
    model_selection:       Dict[str, Any]   # LLM Call 1: primary + secondary + grids
    primary_results:       Dict[str, Any]   # training results for primary model
    secondary_results:     Dict[str, Any]   # training results for secondary model
    evaluation_decision:   Dict[str, Any]   # LLM Call 2: winner + narrative + next_action
    modelling_log:         List[str]


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(user_payload: Dict) -> Dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SKILL_PROMPT},
            {"role": "user",   "content": json.dumps(user_payload, indent=2)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)
    print(f"  [LLM raw response] {parsed}")
    return parsed


# ── Context Builders ──────────────────────────────────────────────────────────

def build_model_selection_context(state: AgentState) -> Dict:
    profile    = state["post_cleaning_profile"]
    class_dist = profile.get("class_distribution", {})
    majority_pct = max(v["pct"] for v in class_dist.values()) if class_dist else None

    ctx = {
        "target_col":         "Severity_Binary",
        "num_rows":           profile.get("num_rows"),
        "num_features":       len(state["feature_cols"]),
        "feature_cols":       state["feature_cols"],
        "class_weight_ratio": profile.get("class_weight_ratio"),
        "majority_class_pct": majority_pct,
        "resampling_note":    "SMOTETomek is applied automatically — class imbalance is handled upstream. Focus model selection on architecture fit.",
        "spark_mode":         "local[*] — keep each model's grid to 4-6 combinations",
        "available_models": {
            name: {
                "task":           info["task"],
                "tunable_params": list(info["tunable"].keys()),
            }
            for name, info in MODEL_REGISTRY.items()
        },
    }

    if state.get("iteration", 0) > 0:
        prev_eval = state.get("evaluation_decision", {})
        ctx["previous_attempt"] = {
            "primary_model":   state["primary_results"].get("model_name"),
            "primary_auc":     state["primary_results"].get("auc"),
            "secondary_model": state["secondary_results"].get("model_name"),
            "secondary_auc":   state["secondary_results"].get("auc"),
            "retrain_guidance": prev_eval.get("retrain_guidance"),
        }

    return ctx


def build_evaluation_context(state: AgentState) -> Dict:
    return {
        "call":             2,
        "iteration":        state.get("iteration", 0),
        "max_iterations":   2,
        "auc_threshold":    0.80,
        "primary_model":    state["primary_results"]["model_name"],
        "primary_auc":      state["primary_results"]["auc"],
        "primary_params":   state["primary_results"].get("best_params", {}),
        "secondary_model":  state["secondary_results"]["model_name"],
        "secondary_auc":    state["secondary_results"]["auc"],
        "secondary_params": state["secondary_results"].get("best_params", {}),
    }


# ── LangGraph Nodes ───────────────────────────────────────────────────────────

def model_selection_node(state: AgentState) -> AgentState:
    print("\n[Modelling Agent] Node: model_selection")

    context  = build_model_selection_context(state)
    decision = call_llm(context)
    state["model_selection"] = decision

    state["modelling_log"] = state.get("modelling_log", []) + [
        f"[model_selection] "
        f"primary={decision['primary_model']} | "
        f"secondary={decision['secondary_model']} | "
        f"excluded={list(decision.get('excluded_models', {}).keys())}"
    ]

    print(f"  Primary   : {decision['primary_model']}")
    print(f"  Secondary : {decision['secondary_model']}")
    print(f"  Why       : {decision['justification']}")
    print(f"  Excluded  : {list(decision.get('excluded_models', {}).keys())}")
    return state


def human_review_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop gate before expensive training.
    Pauses with interrupt() until caller resumes with Command(resume={...}).

    Resume options:
      {"approved": True}
      {"approved": False, "override_primary": "ModelName", "override_secondary": "ModelName"}
    """
    decision = state["model_selection"]

    human_input = interrupt({
        "primary_model":        decision["primary_model"],
        "primary_param_grid":   decision.get("primary_param_grid"),
        "secondary_model":      decision["secondary_model"],
        "secondary_param_grid": decision.get("secondary_param_grid"),
        "justification":        decision["justification"],
        "param_grid_rationale": decision.get("param_grid_rationale"),
        "excluded_models":      decision.get("excluded_models", {}),
    })

    approved           = human_input.get("approved", True)
    override_primary   = human_input.get("override_primary", "").strip()
    override_secondary = human_input.get("override_secondary", "").strip()

    if not approved:
        if override_primary and override_primary in MODEL_REGISTRY:
            state["model_selection"]["primary_model"]    = override_primary
            state["model_selection"]["primary_param_grid"] = MODEL_REGISTRY[override_primary]["tunable"]
            print(f"  Primary overridden to {override_primary} — using registry default grid")
        elif override_primary:
            print(f"  '{override_primary}' not in registry. Keeping agent's primary selection.")

        if override_secondary and override_secondary in MODEL_REGISTRY:
            state["model_selection"]["secondary_model"]    = override_secondary
            state["model_selection"]["secondary_param_grid"] = MODEL_REGISTRY[override_secondary]["tunable"]
            print(f"  Secondary overridden to {override_secondary} — using registry default grid")
        elif override_secondary:
            print(f"  '{override_secondary}' not in registry. Keeping agent's secondary selection.")

    state["modelling_log"] = state["modelling_log"] + [
        f"[human_review] approved={approved} | "
        f"primary={state['model_selection']['primary_model']} | "
        f"secondary={state['model_selection']['secondary_model']}"
    ]
    return state


def training_node(state: AgentState) -> AgentState:
    """Trains both models sequentially using run_training() from Modelling_Skills.py."""
    decision = state["model_selection"]

    for role in ("primary", "secondary"):
        model_name = decision[f"{role}_model"]
        param_grid = decision.get(f"{role}_param_grid", {})

        print(f"\n[Modelling Agent] Node: training | {role}={model_name}")

        results = run_training(
            df=engineered_df,
            model_name=model_name,
            param_grid_spec=param_grid,
            feature_cols=state["feature_cols"],
        )
        state[f"{role}_results"] = results

        state["modelling_log"] = state["modelling_log"] + [
            f"[training_{role}] model={model_name} | "
            f"auc={results['auc']} | "
            f"best_params={results['best_params']}"
        ]

        if results.get("feature_importances"):
            top5 = list(results["feature_importances"].items())[:5]
            print(f"  Top-5 feats : {top5}")

    return state


def evaluation_node(state: AgentState) -> AgentState:
    """LLM Call 2. Compares both AUC results, picks winner, generates narrative."""
    print("\n[Modelling Agent] Node: evaluation")

    context  = build_evaluation_context(state)
    decision = call_llm(context)
    state["evaluation_decision"] = decision
    state["iteration"] = state.get("iteration", 0) + 1

    state["modelling_log"] = state["modelling_log"] + [
        f"[evaluation] winner={decision['winner']} | "
        f"winner_auc={decision['winner_auc']} | "
        f"runner_up={decision['runner_up']} | "
        f"runner_up_auc={decision['runner_up_auc']} | "
        f"next_action={decision.get('next_action')}"
    ]

    print(f"  Winner      : {decision['winner']} (AUC {decision['winner_auc']})")
    print(f"  Runner-up   : {decision['runner_up']} (AUC {decision['runner_up_auc']})")
    print(f"  Next action : {decision.get('next_action')}")
    if decision.get("next_action") == "retrain":
        print(f"  Guidance    : {decision.get('retrain_guidance')}")
    print(f"  Narrative   : {decision['narrative']}")
    return state


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_evaluation(state: AgentState) -> str:
    decision  = state.get("evaluation_decision", {})
    iteration = state.get("iteration", 0)
    if decision.get("next_action") == "retrain" and iteration < 2:
        print(f"\n  [Router] Retrying — iteration {iteration}/2")
        return "model_selection_node"
    print(f"\n  [Router] Accepting results after iteration {iteration}")
    return END


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
    {"model_selection_node": "model_selection_node", END: END},
)

checkpointer    = MemorySaver()
modelling_agent = workflow.compile(checkpointer=checkpointer)
print("Modelling agent graph compiled.")


# ── Run ───────────────────────────────────────────────────────────────────────
# Load your engineered parquet and pass it in as df below.

from pyspark.sql import SparkSession
from pathlib import Path
import psutil
import sys

cores = os.cpu_count()

total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
driver_memory_gb = int(total_ram_gb * 0.7)
driver_memory = f"{driver_memory_gb}g"

print(f"Cores: {cores}")
print(f"Driver Memory: {driver_memory}")

PYTHON_PATH = sys.executable

os.environ['PYSPARK_PYTHON']        = PYTHON_PATH
os.environ['PYSPARK_DRIVER_PYTHON'] = PYTHON_PATH
os.environ['SPARK_LOCAL_IP']        = '127.0.0.1'
os.environ['PYSPARK_PIN_THREAD']    = 'true'

print("Python path:", PYTHON_PATH)
print("Python version:", sys.version)

if "spark" in locals():
    spark.stop()
    print("Existing Spark session stopped.")

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SparkTest") \
    .config("spark.driver.host",                       "127.0.0.1") \
    .config("spark.driver.bindAddress",                "127.0.0.1") \
    .config("spark.driver.memory",                     driver_memory) \
    .config("spark.sql.shuffle.partitions",            cores * 3) \
    .config("spark.default.parallelism",               cores) \
    .config("spark.sql.adaptive.enabled",              "true") \
    .config("spark.pyspark.python",                    PYTHON_PATH) \
    .config("spark.pyspark.driver.python",             PYTHON_PATH) \
    .config("spark.python.use.daemon",                 "false") \
    .config("spark.python.worker.faulthandler.enabled","true") \
    .config("spark.driver.extraJavaOptions",           "-Djava.net.preferIPv4Stack=true") \
    .config("spark.executor.extraJavaOptions",         "-Djava.net.preferIPv4Stack=true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

PROJECT_ROOT = Path.cwd()
ENGINEERED_PARQUET_PATH = PROJECT_ROOT / "dataset/engineered_df.parquet"

if not ENGINEERED_PARQUET_PATH.exists():
    raise FileNotFoundError(f"Engineered Parquet not found at: {ENGINEERED_PARQUET_PATH}")

print("Project root:", PROJECT_ROOT)
print("Original Parquet:", ENGINEERED_PARQUET_PATH)

engineered_df = spark.read.parquet(str(ENGINEERED_PARQUET_PATH))
print(f"Loaded: {engineered_df.count():,} rows")

initial_state: AgentState = {
    "iteration":    0,
    "feature_cols": [                       # <-- from FE final_state
        "is_rush_hour", "Weather_Condition_idx", "Wind_Direction_ohe",
        "State_ohe", "Sunrise_Sunset_ohe", "Start_Time_Hour",
        "Start_Time_is_Weekend", "Visibility(mi)_bin", "Temperature(F)_bin",
        "Wind_Speed(mph)_bin", "Distance(mi)", "Duration_Minutes",
        "Distance(mi)_ratio_Duration_Minutes",
    ],
    "post_cleaning_profile": {              # <-- from FE final_state
        "num_rows": 5270673,
        "class_distribution": {
            0: {"count": 4531863, "pct": 85.98},
            1: {"count": 738810,  "pct": 14.02},
        },
        "class_weight_ratio": 6.13,
    },

    "model_selection":     {},
    "primary_results":     {},
    "secondary_results":   {},
    "evaluation_decision": {},
    "modelling_log":       [],
}

print("\nStarting Modelling Agent...\n")
config = {"configurable": {"thread_id": "modelling_run_3"}}

modelling_agent.invoke(initial_state, config=config)

# Handle human_review interrupt
while modelling_agent.get_state(config).next:
    snapshot      = modelling_agent.get_state(config)
    interrupt_val = snapshot.tasks[0].interrupts[0].value

    print("\n" + "=" * 60)
    print("HUMAN REVIEW -- Model Selection")
    print("=" * 60)
    print(f"  Primary   : {interrupt_val['primary_model']}")
    print(f"  Grid      : {json.dumps(interrupt_val.get('primary_param_grid'), indent=4)}")
    print(f"  Secondary : {interrupt_val['secondary_model']}")
    print(f"  Grid      : {json.dumps(interrupt_val.get('secondary_param_grid'), indent=4)}")
    print(f"  Why       : {interrupt_val['justification']}")
    print(f"  Grid note : {interrupt_val.get('param_grid_rationale')}")
    print("\n  Excluded models:")
    for model, reason in interrupt_val.get("excluded_models", {}).items():
        print(f"    {model}: {reason}")
    print("=" * 60)

    approval = input("\nApprove? (y to proceed, n to override): ").strip().lower()

    if approval == "y":
        resume = {"approved": True}
    else:
        op = input("Override primary model (or press Enter to keep): ").strip()
        os_ = input("Override secondary model (or press Enter to keep): ").strip()
        resume = {"approved": False, "override_primary": op, "override_secondary": os_}

    modelling_agent.invoke(Command(resume=resume), config=config)

final_state = modelling_agent.get_state(config).values


# ── Print Results ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MODELLING AGENT COMPLETE")
print("=" * 60)

ev = final_state["evaluation_decision"]
print(f"Winner    : {ev.get('winner')} (AUC {ev.get('winner_auc')})")
print(f"Runner-up : {ev.get('runner_up')} (AUC {ev.get('runner_up_auc')})")
print(f"\nNarrative (for report):\n{ev.get('narrative')}")

print("\nFull audit log:")
for entry in final_state["modelling_log"]:
    print(f"  {entry}")

for role in ("primary", "secondary"):
    results = final_state[f"{role}_results"]
    if results.get("feature_importances"):
        print(f"\nTop-10 feature importances ({results['model_name']}):")
        for feat, imp in list(results["feature_importances"].items())[:10]:
            print(f"  {feat:<45} {imp:.4f}")
