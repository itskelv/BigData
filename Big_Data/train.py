import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class SentimentTrainer:
    def __init__(self, model_name, num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def tokenize_data(self, texts, max_length=256):
        """Tokenize text data"""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def create_dataset(self, encodings, labels):
        """Create PyTorch dataset"""
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
        
        return ModelDataset(encodings, labels)
    
    def compute_metrics(self, p):
        """Compute evaluation metrics"""
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
    
    def train(self, train_dataset, val_dataset, training_args):
        """Train the model"""
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        print("Starting training...")
        trainer.train()
        return trainer
    
    def save_model(self, trainer, save_path):
        """Save the trained model"""
        trainer.save_model(save_path)
        print(f"Model saved to {save_path}")