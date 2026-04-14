---
name: skill_bin_numeric.md
description: Discretise a continuous numeric column into ordered integer bins using known thresholds
mode: orgnisational
---

# skill_bin_numeric.md

## Purpose

Discretise a continuous numeric column into ordered integer bins.
Useful when a numeric feature has known meaningful thresholds that
a tree model might not discover efficiently on its own.

## When the agent should call this skill

Apply this skill to continuous numeric columns where domain knowledge dictates meaningful thresholds (e.g., freezing temperatures, visibility limits). You are expected to apply this to highly sensitive environmental factors like Temperature(F) and Visibility(mi). 

## Recommended candidates for this dataset

| Column          | Suggested cuts   | Reasoning                        |
| --------------- | ---------------- | -------------------------------- |
| Visibility(mi)  | [0.25, 1.0, 5.0] | Near-zero, poor, moderate, clear |
| Temperature(F)  | [32, 50, 80]     | Freezing, cold, mild, hot        |
| Wind_Speed(mph) | [15, 30]         | Calm, breezy, high wind          |

## Available strategies

| Strategy    | When to use                           | Required params   |
| ----------- | ------------------------------------- | ----------------- |
| equal_width | No strong domain knowledge about cuts | n_bins            |
| custom_cuts | Meaningful thresholds known from EDA  | cuts: list[float] |

## Input

- df: PySpark DataFrame
- column: str
- strategy: "equal_width" or "custom_cuts"
- n_bins: int (equal_width only)
- cuts: list[float] (custom_cuts only)

## Output

DataFrame with new {column}_bin integer column added.
Original column is retained, agent decides whether to keep or drop it
by including/excluding both in assembler_cols.

## Constraints

- Do not bin is_weekend, is_rush_hour, hour these are already
  low-cardinality integers, binning adds no value
- Output column name for assembler_cols: {column}_bin