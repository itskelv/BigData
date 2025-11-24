from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, concat_ws, regexp_replace, trim, when, length
from pyspark.sql.types import StringType, ArrayType, IntegerType
import re
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
import sys
def main():
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
        .csv("../MARC_dataset/train.csv")
    df_test = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("../MARC_dataset/test.csv")
    df_val = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("../MARC_dataset/validation.csv")
    # adjust star label fortransformer model
    df_train = df_train.withColumn("label", col("stars") - 1)
    df_val = df_val.withColumn("label", col("stars") - 1)
    df_test = df_test.withColumn("label", col("stars") - 1)
    
    df_train.printSchema()
    df_train.show(5)
    print("Data types:")
    df_train.printSchema()
    print("Null counts:")
    from pyspark.sql.functions import sum as spark_sum
    df_train.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df_train.columns]).show()
    print("Sample of review_body:")
    df_train.select("review_body").show(10, truncate=50)
    print("Language distribution:")
    df_train.groupBy("language").count().show()
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
    df_train = df_train.dropna(subset=["stars", "review_body"]) \
                                .drop("review_id", "product_id", "reviewer_id", "product_category", "_c0")
    df_val = df_val.dropna(subset=["stars", "review_body"]) \
                            .drop("review_id", "product_id", "reviewer_id", "product_category", "_c0")
    df_test = df_test.dropna(subset=["stars", "review_body"]) \
                                .drop("review_id", "product_id", "reviewer_id", "product_category", "_c0")
    
    # show cleaned text diffrence
    # df_train.select("review_body", "clean_text", "stars").show(5, truncate=100)
    # TOKENIZATION
    # Convert required features to Pandas for tokenization
    train_pd = df_train.select("clean_text", "label").toPandas()
    val_pd = df_val.select("clean_text", "label").toPandas()
    print(f"Training samples: {len(train_pd)}")
    print(f"Validation samples: {len(val_pd)}")
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    # Tokenize the data
    def tokenize_function(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    print("Tokenizing training data...")
    tokenized_train = tokenize_function(train_pd["clean_text"].tolist())
    tokenized_val = tokenize_function(val_pd["clean_text"].tolist())
    # check tokenized data
    train_labels = (train_pd["label"]).tolist()
    # print("=== Tokenized Data Samples ===")
    # for i in range(3):
    #     print(f"\n--- Sample {i+1} ---")
    #     print(f"Original text: {train_pd['clean_text'].iloc[i][:100]}...")
    #     print(f"Label: {train_labels[i]}")
    #     print(f"Input IDs length: {len(tokenized_train['input_ids'][i])}")
    #     print(f"Input IDs (first 20): {tokenized_train['input_ids'][i][:20]}")
    #     print(f"Attention mask (first 20): {tokenized_train['attention_mask'][i][:20]}")
        
    #     # Decode back to see tokens
    #     tokens = tokenizer.convert_ids_to_tokens(tokenized_train['input_ids'][i][:20])
    #     print(f"Tokens (first 20): {tokens}")
    # Load XLMR model
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base', 
        num_labels=5
    )
    # utilizing cuda
    print("Using device:", model.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Create PyTorch dataset
    class ModelDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    val_labels = (val_pd["label"]).tolist()
    train_dataset = ModelDataset(tokenized_train, train_labels)
    val_dataset = ModelDataset(tokenized_val, val_labels)
    
    # Define metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'macro_f1': f1_score(labels, predictions, average='macro'),
            'macro_precision': precision_score(labels, predictions, average='macro'),
            'macro_recall': recall_score(labels, predictions, average='macro'),
            'weighted_f1': f1_score(labels, predictions, average='weighted'),
            'weighted_precision': precision_score(labels, predictions, average='weighted'),
            'weighted_recall': recall_score(labels, predictions, average='weighted')
        }
    # Creating training arguments
    training_args = TrainingArguments(
        output_dir='./xlmr-sentiment-quick',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    # Creating training session
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Start training
    print("training...")
    trainer.train()
    # Save the model
    trainer.save_model('./xlmr-sentiment-final')
    print("Model saved!")
     
if __name__ == "__main__":
    main()