---
name: skill_scale_numeric
description: Scale continuous numeric columns to a common range for distance based or gradient based models
mode: organisational
---

# skill_scale_numeric.md

## Purpose

Scale continuous numeric columns to a common range. Required for
distance-based or gradient-based models.

## When the agent should call this skill

ALWAYS call this skill.
Since Logistic Regression is being used as the baseline model, continuous features MUST be scaled for the model to converge properly and provide valid coefficients. While advanced models like RandomForest and GBT are scale-invariant, feeding them scaled data does not harm their performance. Applying scaling here ensures the final dataset is universally compatible across all our planned model pipelines.

## Available strategies

| Strategy | Formula                 | When to use                        |
| -------- | ----------------------- | ---------------------------------- |
| standard | (x - mean) / stddev     | Default for logistic regression    |
| minmax   | (x - min) / (max - min) | When bounded [0,1] range is needed |

## Columns to scale (if scaling is applied)

Apply to all continuous numeric columns together:
Temperature(F), Humidity(%), Pressure(in), Visibility(mi), Wind_Speed(mph)

## Do NOT scale

- Binary columns (is_weekend, is_rush_hour, Sunrise_Sunset)
- Boolean columns (is_rain, is_fog etc.)
- One-hot encoded columns ({col}_ohe)
- Integer index columns ({col}_idx)

## Input

- df: PySpark DataFrame
- columns: list[str] — numeric columns to scale
- strategy: "standard" or "minmax"

## Output

DataFrame with scaled columns replacing originals (same column names).