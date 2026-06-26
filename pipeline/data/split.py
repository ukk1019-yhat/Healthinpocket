import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple


def stratified_split(
    df: pd.DataFrame,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["diagnosis"]
    train_val, test = train_test_split(
        df, test_size=test_split, stratify=y, random_state=seed
    )
    val_frac = val_split / (1 - test_split)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val["diagnosis"], random_state=seed
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path = "data",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(
        f"Splits saved: train={len(train)}, val={len(val)}, test={len(test)}"
    )
