#!/usr/bin/env python3
"""
Export trained PyTorch checkpoint to ONNX format.

Usage:
    python pipeline/export_model.py
    python pipeline/export_model.py --checkpoint outputs/checkpoints/model_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config.loader import load_config
from pipeline.models.create import build_model


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/model_best.pt")
    parser.add_argument("--output", type=str, default="outputs/exports/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")

    model = build_model(**cfg["model"], pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_size = cfg["data"]["img_size"]
    dummy = torch.randn(1, 3, img_size, img_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=args.opset,
    )
    print(f"ONNX model exported to {output_path}")


if __name__ == "__main__":
    main()
