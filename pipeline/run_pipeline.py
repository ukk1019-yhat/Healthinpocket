#!/usr/bin/env python3
"""
End-to-end training pipeline for RetinaScreen AI.

Usage:
    python pipeline/run_pipeline.py                  # uses default config
    python pipeline/run_pipeline.py --config custom.yaml
    python pipeline/run_pipeline.py --download-only    # just download & validate data
    python pipeline/run_pipeline.py --eval-only        # evaluate saved checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config.loader import load_config
from pipeline.data.download import prepare_data_source
from pipeline.data.validate import validate_dataset, compute_dataset_stats
from pipeline.data.split import stratified_split, save_splits
from pipeline.preprocessing.augmentations import get_train_transform, get_eval_transform, RetinopathyDataset
from pipeline.training.trainer import Trainer
from pipeline.training.metrics import compute_metrics, print_metrics
from pipeline.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_label_distribution,
)
from pipeline.models.create import build_model


def main():
    parser = argparse.ArgumentParser(description="RetinaScreen Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--download-only", action="store_true", help="Download and validate only")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate saved checkpoint only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: data source = {cfg['data']['source']}")

    # Step 1: Data ingestion
    print("\n=== Step 1: Data Ingestion ===")
    data_dir = prepare_data_source(cfg)

    label_file = Path(cfg["data"]["label_file"])
    if not label_file.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            label_file = csv_files[0]
            print(f"Using label file: {label_file}")
        else:
            raise FileNotFoundError(f"No CSV found in {data_dir}")

    df = pd.read_csv(label_file)
    print(f"Loaded {len(df)} records from {label_file}")

    # Step 2: Data validation
    print("\n=== Step 2: Data Validation ===")
    df, report = validate_dataset(data_dir, df)
    stats = compute_dataset_stats(df, data_dir)
    print(f"Label distribution: {stats['label_distribution']}")
    plot_label_distribution(df)

    if args.download_only:
        print("Download-only mode. Exiting.")
        return

    # Step 3: Train/Val/Test split
    print("\n=== Step 3: Data Splitting ===")
    train_df, val_df, test_df = stratified_split(
        df,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg["data"]["seed"],
    )
    save_splits(train_df, val_df, test_df)

    # Step 4: Create datasets & loaders
    print("\n=== Step 4: Data Loaders ===")
    train_transform = get_train_transform(cfg)
    eval_transform = get_eval_transform(cfg)

    train_dataset = RetinopathyDataset(train_df, data_dir, train_transform)
    val_dataset = RetinopathyDataset(val_df, data_dir, eval_transform)
    test_dataset = RetinopathyDataset(test_df, data_dir, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["workers"],
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    if args.eval_only:
        print("\n=== Evaluation Mode ===")
        checkpoint_path = Path("outputs/checkpoints/model_best.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = build_model(**cfg["model"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        from pipeline.training.trainer import Trainer
        t = Trainer(cfg, device)
        t.model = model
        test_loss, test_metrics = t.evaluate(test_loader)
        print_metrics(test_metrics, "Test ")
        plot_confusion_matrix(test_metrics["confusion_matrix"])
        with open("outputs/test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print("Evaluation complete. Metrics saved to outputs/test_metrics.json")
        return

    # Step 5: Training
    print("\n=== Step 5: Training ===")
    trainer = Trainer(cfg, device)
    history = trainer.fit(train_loader, val_loader)

    # Step 6: Final evaluation
    print("\n=== Step 6: Evaluation ===")
    val_loss, val_metrics = trainer.evaluate(val_loader)
    print("\n--- Validation Metrics ---")
    print_metrics(val_metrics)

    test_loss, test_metrics = trainer.evaluate(test_loader)
    print("\n--- Test Metrics ---")
    print_metrics(test_metrics)

    # Step 7: Visualizations
    print("\n=== Step 7: Visualizations ===")
    plot_training_history(history)
    plot_confusion_matrix(test_metrics["confusion_matrix"])

    # Step 8: Save results
    results = {
        "best_val_accuracy": trainer.best_metric,
        "test_metrics": test_metrics,
        "cfg": {k: v for k, v in cfg.items() if k != "data"},
    }
    with open("outputs/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to outputs/training_results.json")
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
