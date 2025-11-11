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


def _find_default_model() -> tuple[Path | None, list[Path]]:
    """Try to auto-locate a reasonable default ONNX model path.

    Search order:
      1) WildLens-Model/exported_models/model.onnx
      2) WildLens-Model/scripts/exported_models/model.onnx
      3) latest runs/*/weights/*.onnx or scripts/**/weights/*.onnx
    Returns (best_path_or_none, attempted_locations_list)
    """
    attempts: list[Path] = []
    scripts_dir = Path(__file__).resolve().parent
    model_root = scripts_dir.parent

    p1 = (model_root / 'exported_models' / 'model.onnx')
    p2 = (scripts_dir / 'exported_models' / 'model.onnx')
    attempts.extend([p1, p2])
    for p in (p1, p2):
        if p.exists():
            return p, attempts

    # Fallback search for recent ONNX under common folders
    candidates: list[Path] = []
    try:
        runs_dir = model_root / 'runs'
        if runs_dir.exists():
            candidates += list(runs_dir.rglob('weights/*.onnx'))
    except Exception:
        pass
    try:
        candidates += list(scripts_dir.rglob('weights/*.onnx'))
    except Exception:
        pass
    if candidates:
        # newest by mtime
        candidates = sorted({c.resolve() for c in candidates if c.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0], attempts

    return None, attempts


def main():
    parser = argparse.ArgumentParser(description='Validate exported ONNX model for WildLens (enhanced checks)')
    parser.add_argument('--model', type=str, default='', help='Path to model.onnx. If omitted, auto-locate in exported_models/.')
    parser.add_argument('--labels', type=str, default='', help='Path to labels.txt. If omitted, assumed next to the model.')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--providers', type=str, default='',
                        help='Comma-separated ORT providers. If omitted, auto-selects CUDA→CPU based on availability.')
    parser.add_argument('--auto-providers', action='store_true', default=True,
                        help='Auto-select providers by availability (CUDAExecutionProvider first, else CPUExecutionProvider).')
    parser.add_argument('--list-providers', action='store_true', help='List available ONNX Runtime providers and exit')
    args = parser.parse_args()

    if args.list_providers:
        print('Available ONNX Runtime providers:')
        try:
            # Use the module-level import to avoid shadowing and scope issues
            print(ort.get_available_providers())
        except Exception as e:
            print(f'Failed to query providers: {e}')
        sys.exit(0)

    # Resolve model path (auto if omitted)
    if args.model:
        model_path = Path(args.model)
    else:
        auto_model, tried = _find_default_model()
        if not auto_model:
            print('ERROR: Could not auto-locate an ONNX model. Tried:')
            for t in tried:
                print('  -', t)
            print('Tips: place model at WildLens-Model/exported_models/model.onnx or pass --model PATH')
            sys.exit(2)
        model_path = auto_model

    # Resolve labels path (default next to model)
    labels_path: Path | None
    if args.labels:
        labels_path = Path(args.labels)
    else:
        labels_path = model_path.parent / 'labels.txt'

    if not model_path.exists():
        print(f'ERROR: ONNX model not found: {model_path}')
        sys.exit(2)

    # Load labels
    labels: list[str] = []
    if labels_path and labels_path.exists():
        labels = load_labels(labels_path)
        print(f'Labels loaded: {len(labels)} classes')
    else:
        print(f'WARN: labels.txt not found at {labels_path}. Continuing without class names...')

    # Check ONNX structure
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print('ONNX structure: OK')

    # Determine providers
    selected_providers: list[str]
    manual = [p.strip() for p in args.providers.split(',') if p.strip()]
    if manual:
        selected_providers = manual
        print(f'Providers (manual): {selected_providers}')
    else:
        # Query available providers with a safety guard
        try:
            avail = ort.get_available_providers()
        except Exception as e:
            print(f'WARN: Failed to query ONNX Runtime providers, defaulting to CPU. Error: {e}')
            avail = ['CPUExecutionProvider']
        if args.auto_providers:
            if 'CUDAExecutionProvider' in avail:
                selected_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                selected_providers = ['CPUExecutionProvider']
        else:
            selected_providers = ['CPUExecutionProvider']
        print(f'Providers (auto from availability {avail}): {selected_providers}')

    # Create an inference session (with automatic CPU fallback on provider errors)
    try:
        session = ort.InferenceSession(str(model_path), providers=selected_providers)
    except Exception as e:
        # If CUDA/TensorRT provider fails to load (e.g., missing DLLs), fall back to CPU automatically
        if selected_providers != ['CPUExecutionProvider']:
            print(f'WARN: Failed to create InferenceSession with providers={selected_providers}: {e}')
            print("      Retrying with providers=['CPUExecutionProvider']...")
            try:
                session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                selected_providers = ['CPUExecutionProvider']
            except Exception as e2:
                print(f'ERROR: CPU fallback also failed: {e2}')
                sys.exit(3)
        else:
            print(f'ERROR: Failed to create InferenceSession with providers={selected_providers}: {e}')
            sys.exit(3)

    print_io(session)

    # Try several dummy sizes to exercise dynamic shapes. If the model has a fixed input size, only test that size.
    # Detect fixed vs dynamic H/W from the model input shape
    try:
        inp0 = session.get_inputs()[0]
        shape = inp0.shape
        hdim = shape[-2] if len(shape) >= 4 else None
        wdim = shape[-1] if len(shape) >= 4 else None
        fixed_hw = None
        if isinstance(hdim, int) and hdim > 0 and isinstance(wdim, int) and wdim > 0:
            if hdim == wdim:
                fixed_hw = hdim
            else:
                # Non-square fixed shape; our dummy generator is square, so warn and use the height as best effort
                print(f'NOTE: Model expects fixed non-square input {hdim}x{wdim}. Validator will use {hdim}x{hdim} for dummy input.')
                fixed_hw = hdim
    except Exception:
        fixed_hw = None

    if fixed_hw:
        test_sizes = [fixed_hw]
        print(f'Detected fixed input size: {fixed_hw}x{fixed_hw}. Skipping dynamic-size tests.')
    else:
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
