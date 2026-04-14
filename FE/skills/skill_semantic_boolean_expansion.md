---
name: skill_semantic_boolean_expansion
description: Expand high cardinality text columns into multiple independent binary indicators
mode: organisational
---

# skill_semantic_boolean_expansion.md

## Purpose

Convert compound, descriptive text columns (eg. Weather_Condition or Street) into multiple independent binary indicators (eg. is_rain, is_highway).

## When the agent should call this skill

When a categorical string column contains overlapping, compound information or has extreme cardinality (eg. >100000 unique values) that is better represented as boolean flag rather than a massive one hot encoded vector or integer index

## Reccommended candiddates for this dataset

- Note these are just examples. You are expected to use your domain knowledge to write regex patterns that capture broad categories\*

**Weather_Condition Examples**
* Rain/Storms: pattern "rain|storm|drizzle|shower|thunder" -> is_rain
* Snow/Ice: pattern "snow|sleet|ice|pellets|squall" -> is_snow
* Fog/Visibility: pattern "fog|haze|mist" -> is_fog

**Street Examples:**
* Interstate / Highway: pattern "i-|hwy|highway|expressway|freeway" -> is_highway
* County / State Roads: pattern "cr-|sr-|county|state" -> is_county_road
* Local Street / Avenue: pattern "st|ave|blvd|rd|road|drive|dr" -> is_local_street

## Input

- df: PySpark DataFrame
- source_column: str (The compound text column, e.g., "Weather_Condition" or "Street")
- target_column: str (The new boolean column name, e.g., "is_highway")
- regex_pattern: str (The regex pattern to search for, in lowercase)

## Output

DataFrame with a new integer column (1 if regex matches, 0 otherwise).

## Constraints
- The executor will automatically lowercase the source column before applying your regex. Write all your regex patterns in strictly lowercase.
- Use simple OR pipes (`|`) for your regex.
- You must add the newly created `target_column` names to `assembler_cols`.
- You must add the original `source_column` to `drop_after_encoding`.
- You MUST create MULTIPLE (at least 10) SemanticExpansionDecisions for EACH high-cardinality column, not just one! For instance, if expanding 'Weather_Condition', create separate decisions for 'is_rain', 'is_snow', 'is_fog', etc.

## CRITICAL: Minimum data support and grouping rules
- **AVOID SPARSITY** Do not create hyper specific features. Your regex patterns MUST target broad categories that are likely to appear in at least '1%' of the dataset
- **NO RARE EVENTS* DO not create isolate flags for rare events as they will result in near zero variance.
- **GROUPING Requirements* If a concept is logically sound but too rare on its own, you MUST combine similar concepts using the OR pipes (`|`). Eg. instead of separate 'is_sleet' and 'is_hail', create a combine 'is_frozen_precip' flag.