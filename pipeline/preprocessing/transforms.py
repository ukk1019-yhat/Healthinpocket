import numpy as np
import cv2
from typing import Tuple


IMG_SIZE = 512
CLIP_LIMIT = 2.0
GRID_SIZE = 8


def load_image(path: str | Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def clahe_equalization(img: np.ndarray, clip_limit: float = CLIP_LIMIT, grid_size: int = GRID_SIZE) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    for i in range(3):
        img[:, :, i] = clahe.apply(img[:, :, i])
    return img


def resize_image(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray, mean=None, std=None) -> np.ndarray:
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32) / 255.0
    return (img - mean) / std


def to_tensor(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, (2, 0, 1)).astype(np.float32)
