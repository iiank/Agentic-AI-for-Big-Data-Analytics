---
name: skill_extract_time_features
description: Decompose the start_time timestamp column into numeric temporal features
mode: organisational
---

# skill_extract_time_features.md

## Purpose

Decompose the Start_Time timestamp column into numeric temporal features
usable by PySpark MLlib. Raw timestamps cannot be consumed by tree models
or logistic regression directly.

## When the agent should call this skill

Always. Start_Time must be decomposed before assembly.
Use when `is_rush_hour` is not yet present

## Available features

The agent selects any subset of the following:

| Feature      | PySpark derivation           | Reasoning               |
| ------------ | ---------------------------- | ----------------------- |
| is_rush_hour | hour in [7-9] or [17-19] → 1 | Peak congestion periods |

## Input

- df: PySpark DataFrame
- features_to_extract: list of feature names from the table above

## Output

DataFrame with new integer columns added.