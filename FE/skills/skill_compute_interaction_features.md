---
name: skill_compute_interactions_features
description: Create new features by combining two existing columns to capture interactions effects
mode: organisational
---

# skill_compute_interaction_features.md

## Purpose

Create new features by combining two existing columns. Captures
compound effects that individual features may not express alone.

## When the agent should call this skill

Only when domain reasoning supports the interaction. Each interaction
adds one column to the feature vector.

## Recommended candidates for this dataset

(Note: These are just examples. Using the exact columns provided in the data profile come up with interaction features with the available operations using your domain knowledge on the combination of features that can affect the severity of an accident.)
| Interaction | Op | Reasoning |
| ------------------------------- | -------- | -------------------------------------------------- |
| hour × is_weekend | multiply | Weekend nights are distinct risk profile |
| Temperature(F) x Humidty(%) | multiply | High temp + high humidity compounds weather stress |
| Distance(mi) x Duration_Minutes | ratio | Captures the speed/impact spread of the accident |

## Available operations

| Op       | Formula                | Use when                         |
| -------- | ---------------------- | -------------------------------- |
| multiply | col_a * col_b          | Both columns contribute jointly  |
| add      | col_a + col_b          | Additive combination makes sense |
| subtract | col_a - col_b          | Subtractive combination makes sense |
| ratio    | col_a / (col_b + 1e-6) | One column normalises the other  |

## Input
- df: PySpark DataFrame
- col_a: str (MUST exist in the Dataframe)
- col_b: str (MUST exist in the Dataframe)
- op: "multiply", "add", "subtract" or "ratio"

## Output
For EVERY interaction feature you generate, you must provide:
- col_a: str (MUST exist in the Dataframe)
- col_b: str (MUST exist in the Dataframe)
- op: "multiply", "add", "subtract", or "ratio"
- rationale: A clear explanation of WHY this interaction impacts accident severity.
- DataFrame with new column named {col_a}_{op}_{col_b}.

## Constraints

- Both input columns must already exist in the DataFrame at time of call
- NEVER invent columns that are not in the data profile
- Output column name for assembler cols: {col_a}_{op}_{col_b} where op is one of "multiply", "add", "subtract" or "ratio"
- Do not interact a column with itself
- Generate at least 3 distinct interaction features. Do not stop at 1.