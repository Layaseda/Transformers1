import os
import warnings
import wandb
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    GPT2Model, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
)
from transformers.integrations import WandbCallback

# Suppress warnings and setup wandb
warnings.simplefilter("ignore", category=FutureWarning)
os.environ["WANDB_PROJECT"] = "gpt2"
os.environ["WANDB_LOG_MODEL"] = "end"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Label distribution for {csv_path}:")
        print(self.data["label"].value_counts(normalize=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["conversation"]
        label = self.data.iloc[idx]["label"]

        # Tokenize the conversation text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# GPT-2 model with classification head for sentiment analysis
class GPT2ForSentiment(nn.Module):
    def __init__(self, num_labels, dropout):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self._is_training = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]
        dropped = self.dropout(hidden) if self._is_training else hidden
        logits = self.classifier(dropped)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def train(self, mode=True):
        super().train(mode)
        self._is_training = mode
        return self

# Function to calculate evaluation metrics 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    wandb.log({
        "confusion_matrix": wandb.Image(fig),
        **metrics
    })
    plt.close(fig)

    return metrics


# Training function
def train():
    wandb.init()
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2ForSentiment(num_labels=3, dropout=config.dropout).to(device)
    model.train()

    full_df = pd.read_csv("data/train.csv")  

    # Split into train and validation sets
    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)

    # Save temporary CSVs for training/validation
    train_df.to_csv("temp_train.csv", index=False)
    val_df.to_csv("temp_val.csv", index=False)

    # Load datasets from temporary CSVs
    train_dataset = SentimentDataset("temp_train.csv", tokenizer, max_length=512)
    val_dataset = SentimentDataset("temp_val.csv", tokenizer, max_length=512)


    # Training arguments for Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="./tmp",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        max_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        logging_strategy="steps",      
        logging_steps=10,              
        seed=42,
        disable_tqdm=True
    )


    # Trainer setup with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.01
        )]
    )

    # Disable W&B ending callback to avoid bugs
    WandbCallback.on_train_end = lambda *args, **kwargs: None

    # Train the model
    trainer.train()
    model.eval()


    # Evaluate on validation set
    val_metrics = trainer.evaluate()
    print("Validation Evaluation:", val_metrics)


    # Test set evaluation
    test_dataset = SentimentDataset("D:/AODTU CLASS/spring 2025/transformers/gpt2/test.csv", tokenizer, max_length=512)
    test_metrics = trainer.evaluate(test_dataset)
    print("Test Evaluation:", test_metrics)

     # Log test metrics to W&B
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})


    # Clean up
    os.remove("temp_train.csv")
    os.remove("temp_val.csv")

    wandb.finish()

# Sweep configuration
sweep_config = {
    "method": "grid",
    "metric": {"name": "macro_f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 5e-5]},
        "dropout": {"values": [0.1, 0.3, 0.4, 0.5]} 
    }
}


# Run the training loop using W&B 
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="gpt2")
    wandb.agent(sweep_id, function=train, count=6)
 

