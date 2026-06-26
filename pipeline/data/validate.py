import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd
from typing import List, Tuple, Optional


MIN_IMG_SIZE = 100
MAX_ASPECT_RATIO = 4.0
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def validate_image(path: Path) -> Optional[str]:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return f"Unsupported extension: {path.suffix}"
    img = cv2.imread(str(path))
    if img is None:
        return f"Corrupt or unreadable: {path.name}"
    h, w = img.shape[:2]
    if h < MIN_IMG_SIZE or w < MIN_IMG_SIZE:
        return f"Too small ({w}x{h}): {path.name}"
    if h == 0 or w == 0 or max(h, w) / min(h, w) > MAX_ASPECT_RATIO:
        return f"Extreme aspect ratio: {path.name}"
    return None


def validate_dataset(img_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    img_dir = Path(img_dir)
    errors = []
    valid_indices = []

    for idx, row in df.iterrows():
        img_path = img_dir / row["image"]
        error = validate_image(img_path)
        if error:
            errors.append({"index": idx, "image": row["image"], "error": error})
        else:
            valid_indices.append(idx)

    report = {
        "total": len(df),
        "valid": len(valid_indices),
        "removed": len(errors),
        "errors": errors,
    }
    if errors:
        print(f"Validation: {report['removed']}/{report['total']} images removed")
        for e in errors[:10]:
            print(f"  - {e['image']}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")

    return df.loc[valid_indices].reset_index(drop=True), report


def compute_dataset_stats(df: pd.DataFrame, img_dir: Path) -> dict:
    img_dir = Path(img_dir)
    sizes = []
    for img_name in df["image"]:
        path = img_dir / img_name
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                sizes.append(img.shape[:2])

    sizes = np.array(sizes)
    label_dist = df["diagnosis"].value_counts().sort_index().to_dict()

    return {
        "num_samples": len(df),
        "label_distribution": label_dist,
        "image_sizes": {
            "min_h": int(sizes[:, 0].min()) if len(sizes) else 0,
            "max_h": int(sizes[:, 0].max()) if len(sizes) else 0,
            "min_w": int(sizes[:, 1].min()) if len(sizes) else 0,
            "max_w": int(sizes[:, 1].max()) if len(sizes) else 0,
            "mean_h": float(sizes[:, 0].mean()) if len(sizes) else 0,
            "mean_w": float(sizes[:, 1].mean()) if len(sizes) else 0,
        },
    }
