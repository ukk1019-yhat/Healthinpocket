import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path
import cv2


ONNX_PATH = Path("data/models/model.onnx")
TEST_IMG_PATH = Path("data/images/test_retina.png")
IMG_SIZE = 512
NUM_CLASSES = 5


def create_dummy_onnx():
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, IMG_SIZE, IMG_SIZE])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, NUM_CLASSES])

    W = helper.make_tensor(
        "W", TensorProto.FLOAT,
        [3 * IMG_SIZE * IMG_SIZE, NUM_CLASSES],
        np.random.randn(3 * IMG_SIZE * IMG_SIZE, NUM_CLASSES).astype(np.float32).tobytes(),
        raw=True,
    )
    B = helper.make_tensor(
        "B", TensorProto.FLOAT,
        [NUM_CLASSES],
        np.random.randn(NUM_CLASSES).astype(np.float32).tobytes(),
        raw=True,
    )

    flatten = helper.make_node("Flatten", ["input"], ["flat"], axis=1)
    matmul = helper.make_node("MatMul", ["flat", "W"], ["logits"])
    add = helper.make_node("Add", ["logits", "B"], ["output"])

    graph = helper.make_graph(
        [flatten, matmul, add],
        "dummy_retinopathy",
        [X],
        [Y],
        [W, B],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, ONNX_PATH)
    print(f"Created dummy ONNX model at {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024:.1f} KB)")


def create_test_image():
    TEST_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(TEST_IMG_PATH), img)
    print(f"Created test image at {TEST_IMG_PATH}")


def verify_inference():
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH))
    dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    output = session.run(None, {"input": dummy_input})
    print(f"ONNX inference OK — output shape: {output[0].shape}, values: {output[0][0]}")


if __name__ == "__main__":
    create_dummy_onnx()
    create_test_image()
    verify_inference()
