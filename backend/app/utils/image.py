import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


IMG_SIZE = 512


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


def preprocess_image(
    path: str | Path, img_size: int = IMG_SIZE
) -> np.ndarray:
    img = load_image(path)
    img = resize_image(img, img_size)
    img = normalize(img)
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)
