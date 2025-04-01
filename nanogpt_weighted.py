import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

os.environ["WANDB_PROJECT"] = "sentiment_analyze"
os.environ["WANDB_LOG_MODEL"] = "end"

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer


# Initialize WandB
wandb.init(project="sentiment_analyze", name="nanoGPT-weighted")


# Define hyperparameters
config = {
    "vocab_size": 50257,
    "max_length": 128,
    "n_layers": 6,
    "n_heads": 8,
    "embed_dim": 512,
    "dropout": 0.1,
    "batch_size": 8,
    "lr": 1e-5,
    "epochs": 10,
    "num_labels": 3
}
wandb.config.update(config)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Dataset class for loading text and labels
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["conversation"]
        label = self.data.iloc[idx]["label"]
        tokens = self.tokenizer(text, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Transformer block with multi-head attention and feedforward layers
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x



# nanoGPT model adapted for classification
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
        x = x[:, -1, :]  # Use the last token's output
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits



# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset = SentimentDataset("D:/AODTU CLASS/spring 2025/transformers/gpt2/train.csv", tokenizer)
val_dataset = SentimentDataset("D:/AODTU CLASS/spring 2025/transformers/gpt2/val.csv", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])



# Compute class weights and boost the positive class
labels = pd.read_csv("D:/AODTU CLASS/spring 2025/transformers/gpt2/train.csv")["label"].values
class_counts = np.bincount(labels)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights[2] *= 2.0  # Boost positive class
normalized_weights = class_weights / class_weights.sum()
weights_tensor = torch.tensor(normalized_weights, dtype=torch.float).to(device)



# Initialize model, optimizer, and loss function
model = NanoGPTClassifier(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    max_length=config["max_length"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    num_labels=config["num_labels"],
    dropout=config["dropout"]
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss(weight=weights_tensor)



# Evaluation function to compute metrics and log confusion matrix
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
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

     # Plot and log confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    wandb.log({
        "confusion_matrix": wandb.Image(fig),
        "val_loss": total_loss / len(dataloader),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    })

    plt.close(fig)
    return total_loss / len(dataloader)


# Training loop
best_val_loss = float("inf")

for epoch in range(config["epochs"]):
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
            wandb.log({
                "train_loss": running_loss / (step + 1),
                "epoch": epoch,
                "step": epoch * len(train_loader) + step
            })

    print(f"Epoch {epoch + 1} | Train Loss: {running_loss / len(train_loader):.4f}")
    val_loss = evaluate(model, val_loader)


    # Save the trained model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "nanogpt-sentiment.pt")
        print("Best model saved.")

print("Training completed.")
