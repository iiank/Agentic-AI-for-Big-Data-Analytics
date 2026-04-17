# BT4221 Project: Understanding US Traffic Accidents (2016-2023)

## Commands to set up connection between Local and Remote Git Repository
```sh
cd insert/your/path/to/the/folder/here
git init
git remote add origin https://github.com/iiank/BT4221_US_Accidents.git
git pull origin main
```

## Register Github account
```sh
git config --global user.email "you@example.com"
git config --global user.name "github username"
```

## Before pushing commits (IMPORTANT!)
### Ensure Local Git Repository is in sync with Remote Git Repository
```sh
git branch -m main          # To ensure you're in main branch
git pull origin main        # Pull any recent commits from other developers 
```

### Pushing commits
```sh
git add .                                              # To stage recent edits
git commit -m "brief description of commit"            # To commit edits with description
git push origin main                                   # Push to remote git repo
```

## Running the project locally (Windows)
### Dependencies
<table>
    <tr>
        <td>Dependency</td>
        <td>Link</td>
    </tr>
    <tr>
        <td>Python</td>
        <td>3.11.15 (Strongly Preferred)</td>
    </tr>
    <tr>
        <td>Java</td>
        <td><a href="https://jdk.java.net//java-se-ri/17-MR1">jdk17.0.0.1</a></td>
    </tr>
    <tr>
        <td>Hadoop (Windows)</td>
        <td><a href="https://hadoop.apache.org/release/3.3.6.html">hadoop-3.3.6.tar.gz</a></td>
    </tr>
    <tr>
        <td>Hadoop bin folder fix (Windows)</td>
        <td><a href="https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6">hadoop-3.3.6 bin fix</a></td>
    </tr>
    <tr>
        <td>WinRAR (Windows)</td>
        <td><a href="https://www.win-rar.com/postdownload.html?&L=0">WinRAR</a></td>
    </tr>
</table>

### Installation process (Windows)
1. Load WinRAR with **administrator rights**, and extract `hadoop-3.3.6.tar.gz` into a folder
2. After extraction, replace the existing bin folder with the one from `Hadoop bin folder fix`

#### Edit Environment Variables
Under the Windows search bar, search for "View advanced system settings". Under "Environment Variables", create 2 new variables under "System Variables": `JAVA_HOME` and `HADOOP_HOME`

`JAVA_HOME`  
```
Variable name:  JAVA_HOME  
Variable value: path/to/jdk/17 (Example: C:\javajdk\jdk-17.0.0.1)
```

`HADOOP_HOME`  
```
Variable name:  HADOOP_HOME  
Variable value: path/to/hadoop/3.3.6 (Example: C:\hadoop\hadoop-3.3.6)
```

Under `Path`, Add 2 new variables: `%JAVA_HOME%\bin` and  `%HADOOP_HOME%\bin`

## Instructions

### Setup
1. Download the project into your preferred directory
2. Confirm the outline of the project shown below:
    ```bash
    C:.
    │   .env
    │   README.md
    |   requirements.txt
    │
    ├───dataset
    │   │   US_Accidents_March23.csv
    │   │
    │   ├───engineered_df_test.parquet
    │   ├───engineered_df_test_pruned.parquet
    │   ├───engineered_df_train.parquet
    │   ├───engineered_df_train_pruned.parquet
    │   ├───test_parquet
    │   ├───train_parquet
    │   └───US_Accidents_March23_parquet
    ├───EDA
    │       EDA_Model.ipynb
    │       EDA_Visuals.ipynb
    │       Visualisations.ipynb
    │
    ├───FE
    │   │   FE_Agent_v4.ipynb
    │   │   ValidateCleaning_Reject.ipynb
    │   │   ValidateCleaning_Skills.py
    │   │
    │   ├───skills
    │   │   │   skill_bin_numeric.md
    │   │   │   skill_compute_interaction_features.md
    │   │   │   skill_drop_columns.md
    │   │   │   skill_encode_categorical.md
    │   │   │   skill_extract_time_features.md
    │   │   │   skill_scale_numeric.md
    │   │   │   skill_semantic_boolean_expansion.md
    │   │   │
    │   │   └───profile-dataset
    │   │           SKILL.md
    │   │
    │   └───state
    │           fe_agent_state.json
    │
    └───Model
            Modelling_Agent.py
            Modelling_Prompt.md
            Modelling_Skills.py
            Model_Evaluator.py
            test_evaluation.py
    ```
3. Enter your OpenAI Secret API key into the __.env__ file

### 1. Data Cleaning and Preparation (Compulsory)
#### Setup
1. Under the __dataset/__ folder, remove all Parquet files. Do not remove the original dataset __US_Accidents_March23.csv__
2. Under the __FE/__ folder, remove the __state/__ folder containing __fe_agent_state.json__

#### Execution
1. Under the __EDA/__ folder, run all cells in __1_EDA_Model.ipynb__
2. 2 new Parquet files will be generated in the __dataset/__ folder:
    - train.parquet
    - test.parquet

### 1.1 Data Analysis and Visualisation (Optional)
#### Execution
1. Under the __EDA/__ folder, run all cells in __1_EDA_Visuals.ipynb__
2. A new Parquet file will be generated in the __dataset/__ folder:
    - visual.parquet
3. After executing __EDA_Visuals.ipynb__, run __Visualisations.ipynb__

### 1.2 Identify Dataset Quality (Optional)
#### Setup
1. Ensure that the __dataset/__ folder consists of:
    - US_Accidents_March23.csv

#### Execution
1. Under the __FE/__ folder, run __ValidateCleaning_Reject.ipynb__ with the 2 Parquet files in the __dataset/__ folder
2. Identifies quality of dataset prior to Feature Engineering Agent

### 2. Validate Cleaning Agent and Feature Engineering Agent (Compulsory)
#### Setup
1. Ensure that the __dataset/__ folder consists of:
    - train.parquet
    - test.parquet

#### Execution
1. After executing __1_EDA_Model.ipynb__, exit the __EDA/__ folder and under the __FE/__ folder, run all cells in __2_VC_FE_Agent.ipynb__
2. A __state/__ folder will be generated in the __FE/__ folder, containing __fe_agent_state.json__
3. 2 new Parquet files will be generated under the __dataset/__ folder:
    - engineered_df_train.parquet
    - engineered_df_test.parquet
4. 2 additional Parquet files may be generated under the __dataset/__ folder if the Feature Engineering Agent decides to prune sparse or noisy features:
    - engineered_df_train_pruned.parquet
    - engineered_df_test_pruned.parquet

### 3. Model & Performance Agent (Compulsory)
#### Setup
1. Ensure that the __dataset/__ folder consists of:
    - engineered_df_train.parquet __OR__ engineered_df_train_pruned.parquet
    - engineered_df_test.parquet __OR__ engineered_df_test_pruned.parquet

#### Execution
1. After executing __2_VC_FE_Agent.ipynb__, exit the __FE/__ folder and under the __Model/__ folder, run __3_Modelling_Agent.py__
2. User will be required to enter inputs:
    ```
    Approve? (y to proceed, n to override):
    Override primary model (or press Enter to keep):
    Override secondary model (or press Enter to keep):
    ```
3. Once the model has been trained, a new folder __saved_models/__ will be generated under the __Model/__ folder
4. The __saved_models/__ folder will contain 2 model folders the user has selected OR left selected by default, in each model folder they contain:
    - The respective model's metadata
    - feature_importances.json
    - results.json

### 4. Evaluation (Compulsory)
#### Setup
1. Ensure that the __Model/__ folder consists of:
    - 2 model folders containing:
        - The respective model's metadata
        - feature_importances.json
        - results.json

#### Execution:
1. After executing __3_Modelling_Agent.py__, run __4_Model_Evaluator.py__
2. A new folder __evaluation_plots/__ will be generated
3. The __evaluation_plots/__ folder will contain:
    - curves_comparison.png
    - saved_models_primary_model_prc_curve.json
    - saved_models_primary_model_roc_curve.json
    - saved_models_secondary_model_prc_curve.json
    - saved_models_secondary_model_roc_curve.json