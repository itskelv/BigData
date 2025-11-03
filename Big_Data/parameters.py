class Config:
    # Data paths
    TRAIN_PATH = "./Big_Data/dataset/train.csv"
    TEST_PATH = "./Big_Data/dataset/test.csv"
    VAL_PATH = "./Big_Data/dataset/validation.csv"
    MODEL_SAVE_PATH = "./xlmr-sentiment-final"
    
    # Spark config
    SPARK_CONFIG = {
        "appName": "AmazonReviewsPreprocessing",
        "spark.driver.memory": "8g",
        "spark.executor.memory": "8g"
    }
    
    # Model config
    MODEL_NAME = 'xlm-roberta-base'
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 3e-5
    
    # Training config
    TRAINING_ARGS = {
        "output_dir": './xlmr-sentiment-quick',
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 3e-5,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "fp16": True
    }