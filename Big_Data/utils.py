import os
import sys

def setup_environment():
    """Setup Python environment for Spark"""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def create_spark_session(config):
    """Create and return Spark session"""
    return SparkSession.builder \
        .appName(config["appName"]) \
        .config("spark.driver.memory", config["spark.driver.memory"]) \
        .config("spark.executor.memory", config["spark.executor.memory"]) \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .getOrCreate()

def analyze_data(df, df_name=""):
    """Perform basic data analysis"""
    print(f"\n=== {df_name} Data Analysis ===")
    df.printSchema()
    print(f"Sample data:")
    df.show(5)
    
    from pyspark.sql.functions import sum as spark_sum, col
    print("Null counts:")
    df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()
    
    print("Sample of review_body:")
    df.select("review_body").show(10, truncate=50)
    
    print("Language distribution:")
    df.groupBy("language").count().show()