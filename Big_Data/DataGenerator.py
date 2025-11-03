from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, concat_ws, regexp_replace, trim, when
from pyspark.sql.types import StringType
import re

class DataPreprocessor:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002700-\U000027BF"  # dingbats
            "]+", flags=re.UNICODE)
    
    def load_data(self, train_path, test_path, val_path):
        """Load and prepare the datasets"""
        df_train = self.spark.read.option("header", "true").option("inferSchema", "true").csv(train_path)
        df_test = self.spark.read.option("header", "true").option("inferSchema", "true").csv(test_path)
        df_val = self.spark.read.option("header", "true").option("inferSchema", "true").csv(val_path)
        
        return df_train, df_test, df_val
    
    def adjust_labels(self, df_train, df_test, df_val):
        """Adjust star ratings to labels (0-4)"""
        df_train = df_train.withColumn("label", col("stars") - 1)
        df_val = df_val.withColumn("label", col("stars") - 1)
        df_test = df_test.withColumn("label", col("stars") - 1)
        return df_train, df_test, df_val
    
    def combine_title_body(self, df_train, df_test, df_val):
        """Combine review title with review body"""
        for df in [df_train, df_test, df_val]:
            df = df.withColumn("review_body", 
                            when(col("review_title").isNotNull(), 
                                concat_ws(" ", col("review_title"), col("review_body")))
                            .otherwise(col("review_body"))) \
                  .drop("review_title")
        return df_train, df_test, df_val
    
    def clean_text(self, df):
        """Clean text data"""
        return df.withColumn("clean_text", 
                            regexp_replace(col("review_body"), "<[^>]+>", " ")) \
                .withColumn("clean_text", 
                            regexp_replace(col("clean_text"), "http\\S+|www\\.\\S+", "")) \
                .withColumn("clean_text", 
                            regexp_replace(col("clean_text"), "[🌀-🗿️😀-🙏Ⓜ-ⓩ✨-🔟]", " ")) \
                .withColumn("clean_text", 
                            regexp_replace(col("clean_text"), "[^a-zA-Z0-9\\s.,!?äöüßÄÖÜáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙ]", " ")) \
                .withColumn("clean_text", 
                            regexp_replace(col("clean_text"), "\\s+", " ")) \
                .withColumn("clean_text", 
                            trim(col("clean_text")))
    
    def remove_irrelevant_features(self, df_train, df_test, df_val):
        """Remove unnecessary columns"""
        columns_to_drop = ["review_id", "product_id", "reviewer_id", "product_category", "_c0"]
        
        for df in [df_train, df_test, df_val]:
            df = df.dropna(subset=["stars", "review_body"]).drop(*columns_to_drop)
        
        return df_train, df_test, df_val
    
    def get_processed_data(self, train_path, test_path, val_path):
        """Complete preprocessing pipeline"""
        df_train, df_test, df_val = self.load_data(train_path, test_path, val_path)
        df_train, df_test, df_val = self.adjust_labels(df_train, df_test, df_val)
        df_train, df_test, df_val = self.combine_title_body(df_train, df_test, df_val)
        df_train, df_test, df_val = self.clean_text(df_train), self.clean_text(df_test), self.clean_text(df_val)
        df_train, df_test, df_val = self.remove_irrelevant_features(df_train, df_test, df_val)
        
        return df_train, df_test, df_val