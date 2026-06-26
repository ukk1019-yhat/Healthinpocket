import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
from preprocess import preprocess_for_training


class RetinopathyDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        img_size: int = 512,
        apply_clahe: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image"]
        label = int(row["diagnosis"])
        image = preprocess_for_training(img_path, self.img_size, self.apply_clahe)
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def build_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    model = models.efficientnet_b4(weights="DEFAULT" if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return {
        "loss": total_loss / len(loader),
        "accuracy": float(accuracy),
        "classification_report": classification_report(
            all_labels, all_preds, output_dict=True
        ),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    NUM_CLASSES = 5

    train_dataset = RetinopathyDataset(
        csv_path="data/train.csv", img_dir="data/images", apply_clahe=True
    )
    val_dataset = RetinopathyDataset(
        csv_path="data/val.csv", img_dir="data/images", apply_clahe=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = build_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), "data/models/best_model.pt")
            print(f"  -> New best model saved (acc: {best_acc:.4f})")

    torch.save(model.state_dict(), "data/models/final_model.pt")
    with open("data/models/training_metrics.json", "w") as f:
        json.dump({"best_accuracy": best_acc}, f)
    print(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
