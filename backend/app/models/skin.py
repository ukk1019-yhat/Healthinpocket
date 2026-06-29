import os
import sys
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto, save
from pathlib import Path


SKIN_CLASS_LABELS = [
    "Benign Nevus",
    "Melanoma",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Squamous Cell Carcinoma",
    "Seborrheic Keratosis",
    "Dermatofibroma",
]

_TMP_MODEL_PATH = Path("/tmp/skin.onnx") if sys.platform != "win32" else Path(os.environ.get("TEMP", "C:\\Windows\\Temp")) / "skin.onnx"

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "models" / "skin.onnx"

MODEL_PATH = Path(os.environ.get("SKIN_MODEL_PATH", str(_DEFAULT_PATH)))


def _create_dummy_onnx(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    IMG_SIZE = 512
    NUM_CLASSES = len(SKIN_CLASS_LABELS)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, IMG_SIZE, IMG_SIZE])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, NUM_CLASSES])
    W = helper.make_tensor(
        "W", TensorProto.FLOAT,
        [3 * IMG_SIZE * IMG_SIZE, NUM_CLASSES],
        np.random.randn(3 * IMG_SIZE * IMG_SIZE, NUM_CLASSES).astype(np.float32).tobytes(),
        raw=True,
    )
    B = helper.make_tensor(
        "B", TensorProto.FLOAT, [NUM_CLASSES],
        np.random.randn(NUM_CLASSES).astype(np.float32).tobytes(), raw=True,
    )
    flatten = helper.make_node("Flatten", ["input"], ["flat"], axis=1)
    matmul = helper.make_node("MatMul", ["flat", "W"], ["logits"])
    add = helper.make_node("Add", ["logits", "B"], ["output"])
    graph = helper.make_graph([flatten, matmul, add], "dummy_skin", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, str(path))
    print(f"Generated dummy ONNX skin model at {path}")


class SkinModel:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.session = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            try:
                _create_dummy_onnx(self.model_path)
            except OSError:
                self.model_path = _TMP_MODEL_PATH
                _create_dummy_onnx(self.model_path)
        available = ort.get_available_providers()
        preferred = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        self.session = ort.InferenceSession(str(self.model_path), providers=preferred)

    def predict(self, input_tensor: np.ndarray) -> dict:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        inputs = {self.session.get_inputs()[0].name: input_tensor}
        outputs = self.session.run(None, inputs)
        probabilities = self._softmax(outputs[0])
        return self._format_predictions(probabilities[0])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _format_predictions(self, probs: np.ndarray) -> dict:
        preds = [
            {
                "class_id": int(i),
                "label": SKIN_CLASS_LABELS[i],
                "confidence": float(probs[i]),
            }
            for i in range(len(SKIN_CLASS_LABELS))
        ]
        preds.sort(key=lambda p: p["confidence"], reverse=True)
        return {
            "predictions": preds,
            "primary_diagnosis": preds[0],
        }

    @property
    def is_loaded(self) -> bool:
        return self.session is not None
