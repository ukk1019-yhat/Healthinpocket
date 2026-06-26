import onnxruntime as ort
import numpy as np
from pathlib import Path


CLASS_LABELS = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "models" / "model.onnx"


class RetinopathyModel:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = model_path
        self.session = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                "Run model/export_onnx.py first."
            )
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
                "label": CLASS_LABELS[i],
                "confidence": float(probs[i]),
            }
            for i in range(len(CLASS_LABELS))
        ]
        preds.sort(key=lambda p: p["confidence"], reverse=True)
        return {
            "predictions": preds,
            "primary_diagnosis": preds[0],
        }

    @property
    def is_loaded(self) -> bool:
        return self.session is not None
