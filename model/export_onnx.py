import torch
import torch.onnx
from train import build_model
from pathlib import Path


MODEL_PATH = "data/models/best_model.pt"
ONNX_PATH = "data/models/model.onnx"
NUM_CLASSES = 5
IMG_SIZE = 512


def export_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX model exported to {ONNX_PATH}")


if __name__ == "__main__":
    export_to_onnx()
