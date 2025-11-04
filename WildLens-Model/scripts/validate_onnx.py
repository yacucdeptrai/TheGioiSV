import argparse
from pathlib import Path
import sys
import onnx
import onnxruntime as ort
import numpy as np


def load_labels(labels_path: Path) -> list[str]:
    with open(labels_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def print_io(session: ort.InferenceSession):
    print('Model I/O:')
    for i, inp in enumerate(session.get_inputs()):
        print(f"  Input[{i}] name={inp.name} shape={inp.shape} type={inp.type}")
    for i, out in enumerate(session.get_outputs()):
        print(f"  Output[{i}] name={out.name} shape={out.shape} type={out.type}")


def check_output_shapes(outputs: list[np.ndarray], num_classes: int):
    # Ultralytics ONNX detection exports typically return a single output of shape (batch, n, 6)
    # where last dim = [x,y,w,h,score,cls] OR (batch, 84, N) in older formats.
    ok = True
    msgs = []
    for i, out in enumerate(outputs):
        if not hasattr(out, 'shape'):
            ok = False
            msgs.append(f'Output[{i}] missing shape attribute')
            continue
        shape = out.shape
        if len(shape) < 2:
            ok = False
            msgs.append(f'Output[{i}] has invalid rank: {shape}')
            continue
        # We can't guarantee exact layout across versions. Perform sanity checks only.
        # Sanity: batch dimension is 1 for our dummy input
        if shape[0] != 1:
            msgs.append(f'Output[{i}] unexpected batch dim (expected 1): {shape}')
        # Sanity: at least 4 dims worth of info downstream
        if np.size(out) == 0:
            ok = False
            msgs.append(f'Output[{i}] has zero-size tensor')
    return ok, msgs


def run_dummy_inference(session: ort.InferenceSession, imgsz: int):
    inp = session.get_inputs()[0]
    # Normalize typical input to 1x3xHxW float32
    h = w = imgsz
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)
    outputs = session.run(None, {inp.name: dummy})
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Validate exported ONNX model for WildLens (enhanced checks)')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--providers', type=str, default='CPUExecutionProvider',
                        help='Comma-separated ORT providers, e.g. CUDAExecutionProvider,CPUExecutionProvider')
    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not model_path.exists():
        print(f'ERROR: ONNX model not found: {model_path}')
        sys.exit(2)
    if not labels_path.exists():
        print(f'ERROR: labels.txt not found: {labels_path}')
        sys.exit(2)

    # Load labels
    labels = load_labels(labels_path)
    print(f'Labels loaded: {len(labels)} classes')

    # Check ONNX structure
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print('ONNX structure: OK')

    # Create an inference session
    providers = [p.strip() for p in args.providers.split(',') if p.strip()]
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        print(f'ERROR: Failed to create InferenceSession with providers={providers}: {e}')
        sys.exit(3)

    print_io(session)

    # Try several dummy sizes to exercise dynamic shapes
    test_sizes = sorted(set([args.imgsz, 320, 480, 640]))
    for size in test_sizes:
        print(f'Running dummy inference at {size}x{size}...')
        try:
            outs = run_dummy_inference(session, size)
        except Exception as e:
            print(f'ERROR: Inference failed at {size}x{size}: {e}')
            sys.exit(4)
        ok, msgs = check_output_shapes(outs, len(labels))
        for m in msgs:
            print('WARN:' if 'unexpected' in m else 'NOTE:', m)
        if not ok:
            print('ERROR: Output shape validation failed.')
            sys.exit(5)
        for i, out in enumerate(outs):
            print(f'  Output[{i}] shape={getattr(out, "shape", None)} dtype={getattr(out, "dtype", None)}')

    print('Validation complete: OK')


if __name__ == '__main__':
    main()
