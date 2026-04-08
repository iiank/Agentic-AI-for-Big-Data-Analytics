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
        <td>Java</td>
        <td><a href="https://jdk.java.net//java-se-ri/17-MR1">jdk17.0.0.1</a></td>
    </tr>
    <tr>
        <td>Hadoop</td>
        <td><a href="https://hadoop.apache.org/release/3.3.6.html">hadoop-3.3.6.tar.gz</a></td>
    </tr>
    <tr>
        <td>Hadoop bin file</td>
        <td><a href="https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin">hadoop-3.3.6 bin</a></td>
    </tr>
</table>

### Edit Environment Variables
Under the Windows search bar, search for "View advanced system settings". Under "Environment Variables", create 2 new variables under "System Variables": `JAVA_HOME` and `HADOOP_HOME`

`JAVA_HOME`  
Variable name:&nbsp;&nbsp;&nbsp;&nbsp;JAVA_HOME  
Variable value:&nbsp;&nbsp;&nbsp;&nbsp;path/to/jdk/17&nbsp;&nbsp;&nbsp;&nbsp;(Example: C:\javajdk\jdk-17.0.0.1)

`HADOOP_HOME`  
Variable name:&nbsp;&nbsp;&nbsp;&nbsp;HADOOP_HOME  
Variable value:&nbsp;&nbsp;&nbsp;&nbsp;path/to/hadoop/3.3.6&nbsp;&nbsp;&nbsp;&nbsp;(Example: C:\hadoop\hadoop-3.3.6)

Under `Path`, Add 2 new variables: %JAVA_HOME%\bin and  %HADOOP_HOME%\bin

## SparkSession
```sh
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("US_Accidents") \
    .config("spark.driver.memory", "") \                // Set the amount of RAM based on availability
    .config("spark.default.parallelism", "") \          // Set based on the number of Logical Processors (Cores * 2)
    .config("spark.sql.shuffle.partitions", "") \       // Whatever you set for spark.default.parallelism * 3
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
```
