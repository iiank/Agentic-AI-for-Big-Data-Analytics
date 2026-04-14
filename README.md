# BT4221_US_Accidents

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
        <td>3.11.* (3.13.9 & 3.14.3 seems to have compatibility issues)</td>
    </tr>
    <tr>
        <td>Java</td>
        <td><a href="https://jdk.java.net//java-se-ri/17-MR1">jdk17.0.0.1</a></td>
    </tr>
    <tr>
        <td>Hadoop</td>
        <td><a href="https://hadoop.apache.org/release/3.3.6.html">hadoop-3.3.6.tar.gz</a></td>
    </tr>
    <tr>
        <td>Hadoop bin folder fix</td>
        <td><a href="https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6">hadoop-3.3.6 bin fix</a></td>
    </tr>
    <tr>
        <td>WinRAR</td>
        <td><a href="https://www.win-rar.com/postdownload.html?&L=0">WinRAR</a></td>
    </tr>
</table>

### Installation process (Windows)
1. Load WinRAR with **administrator rights**, and extract `hadoop-3.3.6.tar.gz` into a folder
2. After extraction, replace the existing bin folder with the one from `Hadoop bin folder fix`

### Edit Environment Variables
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

## Problem Statement and Dataset
Traffic congestion and road safety are critical issues for urban planning and emergency response management. This project leverages a countrywide dataset of US traffic accidents, spanning 49 states and collected via real-time traffic APIs from 2016 to 2023. The primary objective is to determine whether the severity of an accident, gauged by its impact on traffic flow, can be accurately predicted using real-time environmental, temporal, and spatial conditions.

The original dataset “US Accidents (2016 - 2023)” dataset is sourced from Kaggle.  It contains data collected from February 2016 to March 2023 on all 49 states in the United States of America. The dataset was obtained using multiple APIs on streaming traffic incident data from various entities, such as state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within road networks. 

The target variable is "Severity", recorded on a scale from 1 to 4 of increasing traffic disruption. 

## Overall Agent, Langgraph, and Agent Skills Integration
...

## Cleaning & Preprocessing
Primary data cleaning and preprocessing on the 4 severity classes from the raw `US Accidents (2016 - 2023)` was conducted in `EDA/EDA.ipynb`. 

...

## EDA 
Exploratory Data Analysis on the 4 severity classes was also conducted in `EDA/EDA.ipynb`.

...

To manage severe class imbalance and focus on identifying factors that lead to significant traffic disruption, the target is reclassified into a binary variable for EDA in `EDA/post_clean_EDA.ipynb`:

*   Low Severity (Class 0): Original levels 1 and 2, representing accidents with minor to moderate impact on traffic flow.
*   High Severity (Class 1): Original levels 3 and 4, representing accidents with significant impact on traffic flow.

## Validate Cleaning Agent

...

## Feature Engineering Agent

...

## Model Agent

...

## Model Evaluation

...

## Results Discussion & Insights

...