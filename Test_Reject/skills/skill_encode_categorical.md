---
name: skill_encode_categorical
description: convert string categorical columns to numeric form compatible with PySpark MLlib. MLlib cannnot consume raw string columns
mode: orgnisational
---

# skill_encode_categorical.md

## Purpose

Convert string categorical columns to numeric form compatible with
PySpark MLlib. MLlib cannot consume raw string columns.

## When the agent should call this skill

For every string column in the cleaned data profile that is not being dropped.

## Available methods

| Method            | When to use                       | Output column     |
| ----------------- | --------------------------------- | ----------------- |
| string_index_ohe  | n_unique <= 20                    | {col}_ohe        |
| string_index_only | n_unique > 20                     | {col}_idx        |
| passthrough       | column is already numeric/boolean | {col} (unchanged) |
| drop              | column has no predictive value    | column removed    |

## Detail

**string_index_ohe**: Applies StringIndexer then OneHotEncoder.
Produces a sparse binary vector. Use when cardinality is low enough
that the resulting dimensions won't explode the feature vector.
Threshold: n_unique <= 20.

**string_index_only**: Applies StringIndexer only. Assigns an integer
index to each category. Use for high-cardinality columns (e.g. City
with 9000+ unique values) where OHE would produce thousands of dimensions.

**passthrough**: Column is already in a numeric form MLlib can consume
(double, int, boolean). No transformation needed. Use for all boolean
columns produced by the weather condition expansion (is_rain, is_fog etc.)
and for columns already binary-encoded.

**drop**: Column carries no signal useful for severity prediction.
Use for free-text columns (Description) and columns made redundant
by derived features.

## Input

- df: PySpark DataFrame
- column: str — column name
- method: one of the four methods above

## Output

DataFrame with encoded column added. Original string column is NOT
dropped here, add it to drop_after_encoding in FeatureEngDecision.

## Assembler col naming

After encoding, use the output column name in assembler_cols:

- string_index_ohe → {col}_ohe
- string_index_only → {col}_idx
- passthrough → {col}

## CRITICAL RULES:
- If a geographical column has extreme cardinality (>100,000 unique values, such as Zipcode), you MUST use the 'drop' action. It is too granular and will cause the downstream Decision Trees to crash or overfit.
- Rely on 'City', 'County', and 'State' for geographic clustering instead.