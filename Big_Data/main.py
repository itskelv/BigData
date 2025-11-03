from utils.helpers import setup_environment, create_spark_session, analyze_data
from data.preprocessing import DataPreprocessor
from models.trainer import SentimentTrainer
from config.settings import Config
from pyspark.sql.functions import col
import pandas as pd

def main():
    # Setup
    setup_environment()
    spark = create_spark_session(Config.SPARK_CONFIG)
    
    # Preprocess data
    preprocessor = DataPreprocessor(spark)
    df_train, df_test, df_val = preprocessor.get_processed_data(
        Config.TRAIN_PATH, Config.TEST_PATH, Config.VAL_PATH
    )
    
    # Analyze data
    analyze_data(df_train, "Training")
    
    # Convert to Pandas for training
    train_pd = df_train.select("clean_text", "label").toPandas()
    val_pd = df_val.select("clean_text", "label").toPandas()
    
    print(f"Training samples: {len(train_pd)}")
    print(f"Validation samples: {len(val_pd)}")
    
    # Initialize trainer
    trainer = SentimentTrainer(Config.MODEL_NAME, num_labels=5)
    
    # Tokenize data
    tokenized_train = trainer.tokenize_data(train_pd["clean_text"].tolist(), Config.MAX_LENGTH)
    tokenized_val = trainer.tokenize_data(val_pd["clean_text"].tolist(), Config.MAX_LENGTH)
    
    # Create datasets
    train_dataset = trainer.create_dataset(tokenized_train, train_pd["label"].tolist())
    val_dataset = trainer.create_dataset(tokenized_val, val_pd["label"].tolist())
    
    # Setup training arguments
    training_args = TrainingArguments(**Config.TRAINING_ARGS)
    
    # Train model
    model_trainer = trainer.train(train_dataset, val_dataset, training_args)
    
    # Save model
    trainer.save_model(model_trainer, Config.MODEL_SAVE_PATH)
    
    spark.stop()

if __name__ == "__main__":
    main()