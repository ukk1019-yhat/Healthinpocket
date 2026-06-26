import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from pipeline.training.metrics import compute_metrics, print_metrics
from pipeline.models.create import build_model


class Trainer:
    def __init__(self, cfg: Dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        tc = cfg["training"]
        self.epochs = tc["epochs"]
        self.clip = tc.get("gradient_clip")
        self.label_smoothing = tc.get("label_smoothing", 0.0)
        self.mixed_precision = tc.get("mixed_precision", False) and device.type == "cuda"
        self.early_stop_patience = tc.get("early_stop_patience")
        self.save_every = cfg["logging"]["save_every_n_epochs"]

        self.model = build_model(**cfg["model"]).to(device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
        )
        self._init_scheduler(tc)
        self.scaler = torch.amp.GradScaler(device.type) if self.mixed_precision else None

        self.checkpoint_dir = Path("outputs/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        self.best_metric = 0.0
        self.early_stop_counter = 0

    def _init_scheduler(self, tc: Dict):
        sched = tc.get("scheduler", "cosine")
        if sched == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif sched == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.1
            )
        elif sched == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        else:
            self.scheduler = None

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler:
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        avg_loss = total_loss / len(loader)
        metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
        return avg_loss, metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            lr = self.optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | Kappa: {val_metrics['kappa']:.4f} | "
                f"LR: {lr:.2e}"
            )

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, "checkpoint")
            if val_metrics["accuracy"] > self.best_metric:
                self.best_metric = val_metrics["accuracy"]
                self._save_checkpoint(epoch, "best")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_patience and self.early_stop_counter >= self.early_stop_patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history

    def _save_checkpoint(self, epoch: int, tag: str):
        path = self.checkpoint_dir / f"model_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "cfg": self.cfg,
        }, path)
