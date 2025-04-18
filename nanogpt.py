import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from transformers import GPT2Tokenizer


# Sweep configuration
sweep_config = {
    'method': 'grid',
    'name': 'nanoGPTT',
    'parameters': {
        'n_layers': {'values': [4, 6]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3]},
        'lr': {'values': [1e-4, 5e-5]},
        'max_length': {'value': 512},
        'n_heads': {'values': [4, 8]},
        'embed_dim': {'values': [128, 256]},
        'batch_size': {'value': 8},
        'epochs': {'value': 50}, 
        'vocab_size': {'value': 50257},
        'num_labels': {'value': 3}
    }
}



# Dataset class to handle the data loading and tokenization process
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["conversation"]
        label = self.data.iloc[idx]["label"]

        # Tokenize the text (pad to max_length, truncate if necessary)
        tokens = self.tokenizer(text, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()

        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network (fully connected)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        # Attention layer followed by residual connection and layer normalization
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x


# nanoGPT classifier
class NanoGPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, n_heads, n_layers, num_labels, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.classifier(x)


# Evaluation 
def evaluate(model, dataloader, config, label="val"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    wandb.log({
        f"{label}_confusion_matrix": wandb.Image(fig),
        f"{label}_loss": total_loss / len(dataloader),
        f"{label}_accuracy": acc,
        f"{label}_precision": precision,
        f"{label}_recall": recall,
        f"{label}_macro_f1": f1
    })
    plt.close(fig)


# Training function for wandb sweep
def train():
    wandb.init()
    config = wandb.config

    # Validate head-embedding compatibility
    if config.embed_dim % config.n_heads != 0:
        print(f"Skipping invalid config: embed_dim={config.embed_dim}, n_heads={config.n_heads}")
        return

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Split the training data each run
    df = pd.read_csv("data/train.csv")[["conversation", "label"]]
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df.to_csv("data/train_split.csv", index=False)
    val_df.to_csv("data/val_split.csv", index=False)


    train_dataset = SentimentDataset("data/train_split.csv", tokenizer, max_length=config.max_length)
    val_dataset = SentimentDataset("data/val_split.csv", tokenizer, max_length=config.max_length)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)


    # Create the model and optimizer
    model = NanoGPTClassifier(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_length=config.max_length,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        num_labels=config.num_labels,
        dropout=config.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # Early Stopping Variables
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0
        for step, batch in enumerate(train_loader):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            if step % 10 == 0:
                wandb.log({"train_loss": running_loss / (step + 1)})

        print(f"Epoch {epoch+1} | Train Loss: {running_loss / len(train_loader):.4f}")
        evaluate(model, val_loader, config, label="val")



        # Get the latest val_loss from wandb history
        try:
            current_val_loss = wandb.run.history._data[-1]["val_loss"]
        except:
            current_val_loss = float("inf")


        # Early stopping logic
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break



    # Final test evaluation
    test_df = pd.read_csv("data/test.csv")[["conversation", "label"]]

    test_df.to_csv("data/test_clean.csv", index=False)

    test_dataset = SentimentDataset("data/test_clean.csv", tokenizer, max_length=config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    print("Evaluating on test set...")
    evaluate(model, test_loader, config, label="test")



# Set device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Register and launch the sweep
sweep_id = wandb.sweep(sweep_config, project="nanoGPTT")
wandb.agent(sweep_id, function=train)