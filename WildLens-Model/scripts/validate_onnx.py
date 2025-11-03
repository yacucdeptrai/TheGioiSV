import argparse
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Validate exported ONNX model for WildLens')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    # Check ONNX structure
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print('ONNX structure: OK')

    # Load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f'Labels loaded: {len(labels)} classes')

    # Create an inference session
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])  # GPU optional
    inputs = session.get_inputs()
    input_shape = inputs[0].shape
    input_name = inputs[0].name
    print(f'Input: {input_name} shape={input_shape}')

    # Make a dummy image (1x3xHxW)
    h = args.imgsz
    w = args.imgsz
    dummy = (np.random.rand(1, 3, h, w).astype(np.float32))

    outputs = session.run(None, {input_name: dummy})
    print('Inference ran successfully.')
    for i, out in enumerate(outputs):
        print(f'  Output[{i}] shape={getattr(out, "shape", None)} dtype={getattr(out, "dtype", None)}')

    print('Validation complete.')


if __name__ == '__main__':
    main()
