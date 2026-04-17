# Trial_Modelling_Agent.py
# Orchestration layer for the modelling agent.
# Defines AgentState, LangGraph nodes, LLM calls, and graph wiring.
# Imports PySpark execution from Modelling_Skills.py.

import json
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from Modelling_Skills import MODEL_REGISTRY, run_training

# Visual helper function (Only for printing purposes)
def border(s):
    print()
    print(f"{'='*20} {s} {'='*20}")

# ── LLM Client ────────────────────────────────────────────────────────────────

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
    modelling_log:         Annotated[List[str], operator.add]


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
        "resampling_note":    "Majority class is downsampled to match minority (50:50) automatically — class imbalance is handled upstream. Focus model selection on architecture fit",
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

def model_selection_node(state: AgentState) -> dict:
    print("\n[Modelling Agent] Node: model_selection")

    context  = build_model_selection_context(state)
    decision = call_llm(context)

    print(f"  Primary   : {decision['primary_model']}")
    print(f"  Secondary : {decision['secondary_model']}")
    print(f"  Why       : {decision['justification']}")
    print(f"  Excluded  : {list(decision.get('excluded_models', {}).keys())}")

    return {
        "model_selection": decision,
        "modelling_log": [
            f"[model_selection] "
            f"primary={decision['primary_model']} | "
            f"secondary={decision['secondary_model']} | "
            f"excluded={list(decision.get('excluded_models', {}).keys())}"
        ],
    }


def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-loop gate before expensive training.
    Pauses with interrupt() until caller resumes with Command(resume={...}).

    Resume options:
      {"approved": True}
      {"approved": False, "override_primary": "ModelName", "override_secondary": "ModelName"}
    """
    decision = dict(state["model_selection"])  # copy — do not mutate state directly

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
            decision["primary_model"]     = override_primary
            decision["primary_param_grid"] = MODEL_REGISTRY[override_primary]["tunable"]
            print(f"  Primary overridden to {override_primary} — using registry default grid")
        elif override_primary:
            print(f"  '{override_primary}' not in registry. Keeping agent's primary selection.")

        if override_secondary and override_secondary in MODEL_REGISTRY:
            decision["secondary_model"]     = override_secondary
            decision["secondary_param_grid"] = MODEL_REGISTRY[override_secondary]["tunable"]
            print(f"  Secondary overridden to {override_secondary} — using registry default grid")
        elif override_secondary:
            print(f"  '{override_secondary}' not in registry. Keeping agent's secondary selection.")

    return {
        "model_selection": decision,
        "modelling_log": [
            f"[human_review] approved={approved} | "
            f"primary={decision['primary_model']} | "
            f"secondary={decision['secondary_model']}"
        ],
    }


def training_node(state: AgentState) -> dict:
    """Trains both models sequentially using run_training() from Modelling_Skills.py."""
    decision    = state["model_selection"]
    updates     = {}
    log_entries = []

    for role in ("primary", "secondary"):
        model_name = decision[f"{role}_model"]
        param_grid = decision.get(f"{role}_param_grid", {})

        print(f"\n[Modelling Agent] Node: training | {role}={model_name}")

        results = run_training(
            train_df=train_df,
            test_df=test_df,
            model_name=model_name,
            param_grid_spec=param_grid,
            feature_cols=state["feature_cols"]
        )
        updates[f"{role}_results"] = results
        log_entries.append(
            f"[training_{role}] model={model_name} | "
            f"auc={results['auc']} | "
            f"best_params={results['best_params']}"
        )

        if results.get("feature_importances"):
            top5 = list(results["feature_importances"].items())[:5]
            print(f"  Top-5 feats : {top5}")

    return {**updates, "modelling_log": log_entries}


def evaluation_node(state: AgentState) -> dict:
    """LLM Call 2. Compares both AUC results, picks winner, generates narrative."""
    print("\n[Modelling Agent] Node: evaluation")

    context  = build_evaluation_context(state)
    decision = call_llm(context)

    print(f"  Winner      : {decision['winner']} (AUC {decision['winner_auc']})")
    print(f"  Runner-up   : {decision['runner_up']} (AUC {decision['runner_up_auc']})")
    print(f"  Next action : {decision.get('next_action')}")
    if decision.get("next_action") == "retrain":
        print(f"  Guidance    : {decision.get('retrain_guidance')}")
    print(f"  Narrative   : {decision['narrative']}")

    return {
        "evaluation_decision": decision,
        "iteration": state.get("iteration", 0) + 1,
        "modelling_log": [
            f"[evaluation] winner={decision['winner']} | "
            f"winner_auc={decision['winner_auc']} | "
            f"runner_up={decision['runner_up']} | "
            f"runner_up_auc={decision['runner_up_auc']} | "
            f"next_action={decision.get('next_action')}"
        ],
    }


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

border("Retrieve local computer specifications")
print(f"Cores: {cores}")
print(f"Driver Memory: {driver_memory}")

PYTHON_PATH = sys.executable

os.environ['PYSPARK_PYTHON']        = PYTHON_PATH
os.environ['PYSPARK_DRIVER_PYTHON'] = PYTHON_PATH
os.environ['SPARK_LOCAL_IP']        = '127.0.0.1'
os.environ['PYSPARK_PIN_THREAD']    = 'true'

border("Python installation check")
print("Python path:", PYTHON_PATH)
print("Python version:", sys.version)
border("Extra Logging")

if "spark" in locals():
    spark.stop()
    print("Existing Spark session stopped.")

spark = (
    SparkSession.builder
    .master("local[*]") \
    .appName("SparkTest") \
    .config("spark.driver.host",                       "127.0.0.1")
    .config("spark.driver.bindAddress",                "127.0.0.1")
    .config("spark.pyspark.python",                    PYTHON_PATH)
    .config("spark.pyspark.driver.python",             PYTHON_PATH)
    .config("spark.driver.extraJavaOptions",           "-Djava.net.preferIPv4Stack=true -Dlog4j.logger.org.apache.spark.storage.BlockManagerStorageEndpoint=FATAL")
    .config("spark.executor.extraJavaOptions",         "-Djava.net.preferIPv4Stack=true")
    .config("spark.driver.memory",                     driver_memory)
    .config("spark.sql.shuffle.partitions",            cores * 3)
    .config("spark.default.parallelism",               cores)
    .config("spark.sql.adaptive.enabled",              "true")
    .config("spark.python.use.daemon",                 "false")
    .config("spark.python.worker.faulthandler.enabled","true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

PROJECT_ROOT = Path.cwd()
TRAIN_PARQUET_PATH = PROJECT_ROOT / "dataset" / "engineered_df_train_pruned.parquet"
TEST_PARQUET_PATH  = PROJECT_ROOT / "dataset" / "engineered_df_test_pruned.parquet"

if not TRAIN_PARQUET_PATH.exists() or not TEST_PARQUET_PATH.exists():
    raise FileNotFoundError(f"One or both engineered Parquet files not found.")

border("Identify directories")
print("Project root:", PROJECT_ROOT)
print("Train Parquet:", TRAIN_PARQUET_PATH)
print("Test Parquet:", TEST_PARQUET_PATH)

train_df = spark.read.parquet(str(TRAIN_PARQUET_PATH))
test_df = spark.read.parquet(str(TEST_PARQUET_PATH))

border("Loading FE parquet")
print(f"Loaded Train Data: {train_df.count():,} rows")
print(f"Loaded Test Data:  {test_df.count():,} rows")

# ── Load FE agent state ───────────────────────────────────────────────────────

FE_STATE_PATH = PROJECT_ROOT / "FE/state" / "fe_agent_state.json"
if not FE_STATE_PATH.exists():
    raise FileNotFoundError(f"FE agent state not found at: {FE_STATE_PATH}")

with open(FE_STATE_PATH) as f:
    fe_state = json.load(f)

# Exclude high_severity — it is derived from the target and causes leakage 
feature_cols = [c for c in fe_state["feature_columns"] if c != "high_severity"]

# Derive binary class distribution from FE stats report
stats      = fe_state["feature_stats_report"]
num_rows   = stats["structural_overview"]["total_rows"]
high_ratio = stats["boolean_and_binary_analysis"]["high_severity"]["true_ratio"]
high_count = round(num_rows * high_ratio)
low_count  = num_rows - high_count
post_cleaning_profile = {
    "num_rows": num_rows,
    "class_distribution": {
        0: {"count": low_count,  "pct": round((1 - high_ratio) * 100, 2)},
        1: {"count": high_count, "pct": round(high_ratio * 100, 2)},
    },
    "class_weight_ratio": round(low_count / high_count, 2),
}

border("FE state loaded")
print(f"Feature cols   : {len(feature_cols)} (excluded high_severity)")
print(f"Num rows       : {num_rows:,}")
print(f"Class ratio    : {post_cleaning_profile['class_weight_ratio']} : 1")

initial_state: AgentState = {
    "iteration":             0,
    "feature_cols":          feature_cols,
    "post_cleaning_profile": post_cleaning_profile,
    "model_selection":       {},
    "primary_results":       {},
    "secondary_results":     {},
    "evaluation_decision":   {},
    "modelling_log":         [],
}

print("\nStarting Modelling Agent...\n")
config = {"configurable": {"thread_id": "modelling_run_4"}}

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

    # ── Print Results ─────────────────────────────────────────────────────────
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