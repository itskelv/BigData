import torch
import pandas as pd
from transformers import RemBertForSequenceClassification, RemBertTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import parameters

params = parameters.params

# Import finalized dataset
train_pd = pd.read_pickle(os.path.join(params["final_dir"], "train.pkl"))
val_pd   = pd.read_pickle(os.path.join(params["final_dir"], "val.pkl"))
test_pd = pd.read_pickle(os.path.join(params["final_dir"], "test.pkl"))

print("training samples:", len(train_pd))
print("validation samples:", len(val_pd))

# Load tokenizer
tokenizer = RemBertTokenizer.from_pretrained('google/rembert')
# Tokenize the data
def tokenize_function(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
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

# Available languages
LANG = {
    "en": 0,
    "de": 1,
    "fr": 2,
    "es": 3,
    "ja": 4,
    "zh": 5
}

# Load XLMR model
model = RemBertForSequenceClassification.from_pretrained(
    'google/rembert', 
    num_labels=5
)
# utilizing cuda
print("Using device:", model.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("using:", device)

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
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
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
trainer.save_model('./rbrt-sentiment-final')
print("Model saved!")