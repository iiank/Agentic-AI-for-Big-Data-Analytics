# BT4221_US_Accidents

## Commands to set up connection between Local and Remote Git Repository
```sh
cd insert/your/path/to/the/folder/here
git init
git remote add origin https://github.com/iiank/BT4221_US_Accidents.git
git pull origin main
```

## Before pushing commits (IMPORTANT!)
### Ensure Local Git Repository is in sync with Remote Git Repository
```sh
git branch -m main
git pull origin main
```

## Base Spark Setup
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
