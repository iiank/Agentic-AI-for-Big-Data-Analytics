---
name: skill_drop_columns
description: Remove columns from the DataFrame after encoding is complete to keep the feature space clean
mode: organisational
---

# skill_drop_columns.md

## Purpose

Remove columns from the DataFrame after encoding is complete.
Keeps the feature space clean before VectorAssembler runs.

## When the agent should call this skill

Always after encoding, original string columns and Start_Time
must be removed. They cannot coexist with their encoded counterparts
in the assembler input without causing duplication.

## What to include in drop_after_encoding

Always drop:

- Timestamps provided in your context (eg Start_Time, End_Time)
- Original string columns after string_index_ohe or string_index_only
  e.g. if State → State_ohe, drop State and State_idx
- Any column explicitly marked action="drop" in column_encodings

Do NOT drop:

- Passthrough columns (already numeric, include directly)
- Boolean columns (is_rain, is_fog etc. — include in assembler_cols)
- The target column Severity

## Input

- df: PySpark DataFrame
- columns: list[str] — columns to remove

## Output

For each dropped column, you must provide:
- column: str
- rationale: str (eg. "Original string column replaced by OHE", "Raw timestamp replaced by extracted features")