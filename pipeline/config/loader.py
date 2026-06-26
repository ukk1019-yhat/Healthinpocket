import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: dict, path: str | Path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
