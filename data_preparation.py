from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, concat_ws, regexp_replace, trim, when, sum
import re
import os
import sys
import pandas as pd

import parameters

def main():
    params = parameters.params

    os.makedirs(params["final_dir"], exist_ok=True)
    
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    spark = SparkSession.builder \
    .appName("AmazonReviewsPreprocessing") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .getOrCreate()
    df_train = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(params['train_root_dir'])
    df_test = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(params['test_root_dir'])
    df_val = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(params['val_root_dir'])
    
    print(df_train.columns)

    df_train.printSchema()
    
    print("Language distribution:")
    df_train.groupBy("language").count().show()

    print("Null counts:")
    df_train.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_train.columns]).show()
    
    # combining review title to review body to gain more sentiment value
    df_train = df_train.withColumn("review_body", 
                                when(col("review_title").isNotNull(), 
                                    concat_ws(" ", col("review_title"), col("review_body")))
                                .otherwise(col("review_body"))) \
                    .drop("review_title")
    df_val = df_val.withColumn("review_body", 
                            when(col("review_title").isNotNull(), 
                                concat_ws(" ", col("review_title"), col("review_body")))
                            .otherwise(col("review_body"))) \
                .drop("review_title")
    df_test = df_test.withColumn("review_body", 
                                when(col("review_title").isNotNull(), 
                                    concat_ws(" ", col("review_title"), col("review_body")))
                                .otherwise(col("review_body"))) \
                    .drop("review_title")
    # lists of emoji to be removed
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # dingbats
        "]+", flags=re.UNICODE)
    # Data cleaning function
    def data_cleaning(df):
        return df.withColumn("clean_text", 
                            # Remove HTML tags
                            regexp_replace(col("review_body"), "<[^>]+>", " ")) \
                .withColumn("clean_text", 
                            # Remove URLs
                            regexp_replace(col("clean_text"), "http\\S+|www\\.\\S+", "")) \
                .withColumn("clean_text", 
                            # Remove common emoji
                            regexp_replace(col("clean_text"), "[🌀-🗿😀-🙏Ⓜ-ⓩ✨-🔟]", " ")) \
                .withColumn("clean_text", 
                            # Remove special characters
                            regexp_replace(col("clean_text"), "[^a-zA-Z0-9\\s.,!?äöüßÄÖÜáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙ]", " ")) \
                .withColumn("clean_text", 
                            # Remove multiple whitespaces
                            regexp_replace(col("clean_text"), "\\s+", " ")) \
                .withColumn("clean_text", 
                            # Trim whitespace
                            trim(col("clean_text")))
    
    df_train = data_cleaning(df_train)
    df_val = data_cleaning(df_val)  
    df_test = data_cleaning(df_test)
    # remove irrelevant features
    df_train = df_train.dropna(subset=["label", "review_body", "language"]) \
                                .drop("stars", "review_id", "product_id", "reviewer_id", "product_category", "_c0")
    df_val = df_val.dropna(subset=["label", "review_body", "language"]) \
                            .drop("stars", "review_id", "product_id", "reviewer_id", "product_category", "_c0")
    df_test = df_test.dropna(subset=["label", "review_body", "language"]) \
                                .drop("stars", "review_id", "product_id", "reviewer_id", "product_category", "_c0")
    
    # adjust star label for transformer model
    df_train = df_train.withColumn("label", col("stars") - 1)
    df_val = df_val.withColumn("label", col("stars") - 1)
    df_test = df_test.withColumn("label", col("stars") - 1)
    
    # show cleaned text diffrence
    # df_train.select("review_body", "clean_text", "stars").show(5, truncate=100)
    
    # Convert required features to Pandas for tokenization
    train_pd = df_train.select("clean_text", "label").toPandas()
    val_pd = df_val.select("clean_text", "label").toPandas()
    test_pd = df_test.select("clean_text", "label").toPandas()

    print(f"Training samples: {len(train_pd)}")
    print(f"Validation samples: {len(val_pd)}")
    print(f"Validation samples: {len(test_pd)}")

    train_pd.to_pickle(os.path.join(params["final_dir"], "train.pkl"))
    val_pd.to_pickle(os.path.join(params["final_dir"], "val.pkl"))
    test_pd.to_pickle(os.path.join(params["final_dir"], "test.pkl"))
    
if __name__ == "__main__":
    main()