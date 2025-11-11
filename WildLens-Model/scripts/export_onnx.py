import argparse
from pathlib import Path
import shutil
import sys

import torch
import yaml
from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


TARGET_CLASSES_30 = [
    'Dog','Cat','Horse','Cow','Sheep','Pig','Chicken','Duck','Bird','Elephant',
    'Lion','Tiger','Bear','Monkey','Deer','Fox','Wolf','Rabbit','Squirrel','Giraffe',
    'Zebra','Kangaroo','Panda','Koala','Raccoon','Penguin','Dolphin','Whale','Turtle','Frog'
]


def save_labels_txt(labels: list[str], out_path: Path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for name in labels:
            f.write(str(name).strip() + "\n")


def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_names_from_yaml(data_yaml: Path) -> list[str]:
    try:
        cfg = load_yaml(data_yaml)
        names = cfg.get('names')
        if isinstance(names, dict) and names:
            try:
                items = sorted(((int(k), v) for k, v in names.items()), key=lambda t: t[0])
                return [str(v).strip() for _, v in items]
            except Exception:
                return [str(v).strip() for v in names.values()]
        if isinstance(names, list) and names:
            return [str(n).strip() for n in names if str(n).strip()]
    except Exception:
        pass
    return TARGET_CLASSES_30


def _find_latest_best_pt() -> Path | None:
    """Auto-discover the most recent best.pt under common training output folders.

    Search order (all recursive):
      - scripts/train*/weights/best.pt
      - runs/*/weights/best.pt
      - scripts/**/weights/best.pt (fallback)
    Returns newest by modification time, or None if not found.
    """
    scripts_dir = Path(__file__).resolve().parent
    model_root = scripts_dir.parent  # WildLens-Model

    candidates: list[Path] = []
    try:
        candidates += list(scripts_dir.glob('train*/weights/best.pt'))
    except Exception:
        pass
    try:
        candidates += list((model_root / 'runs').rglob('weights/best.pt')) if (model_root / 'runs').exists() else []
    except Exception:
        pass
    try:
        # broad fallback but still within scripts
        candidates += list(scripts_dir.rglob('weights/best.pt'))
    except Exception:
        pass

    if not candidates:
        return None
    # pick the most recently modified
    candidates = sorted(set(p.resolve() for p in candidates if p.exists()), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained Ultralytics YOLO model (.pt) to ONNX reliably, placing the file exactly where you want it.")
    parser.add_argument('--weights', type=str, required=False,
                        help='Path to best.pt (trained model weights). If omitted, auto-detect the latest train*/runs best.pt')
    parser.add_argument('--out', type=str, default=str(Path(__file__).resolve().parent / 'exported_models' / 'model.onnx'),
                        help='Target ONNX output path (default: scripts/exported_models/model.onnx)')
    parser.add_argument('--imgsz', type=int, default=320, help='Inference image size used for export')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='Run onnx-simplifier after export')
    parser.add_argument('--dynamic', action='store_true', help='Export with dynamic axes')
    parser.add_argument('--optimize', action='store_true', help='Optimize ONNX (requires CPU export)')
    parser.add_argument('--device', type=str, default='cpu', help="Export device: 'cpu' recommended for optimize=True")
    parser.add_argument('--labels-out', type=str, default='',
                        help='Optional path to write labels.txt. Default: alongside --out as labels.txt')
    parser.add_argument('--data', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'data.yaml'),
                        help='Path to data.yaml used as fallback for class names if not embedded in the model')
    parser.add_argument('--show-found', action='store_true', help='Print which checkpoint was auto-selected')
    args = parser.parse_args()

    # Resolve weights: accept file path, directory containing best.pt, or auto-detect if omitted
    weights: Path | None = None
    if args.weights:
        w = Path(args.weights).resolve()
        if w.is_dir():
            wp = w / 'best.pt'
            if wp.exists():
                weights = wp
        elif w.suffix.lower() == '.pt' and w.exists():
            weights = w
    else:
        weights = _find_latest_best_pt()

    if not weights or not weights.exists():
        print("ERROR: Could not determine weights to export.")
        print("Tips:")
        print("  - Pass --weights PATH/TO/best.pt")
        print("  - Or place your checkpoint under scripts/train*/weights/best.pt or runs/*/weights/best.pt for auto-detect.")
        sys.exit(2)
    elif args.show_found:
        print(f"Auto-selected weights: {weights}")

    out_path = Path(args.out).resolve()
    ensure_dir(out_path.parent)
    labels_out = Path(args.labels_out).resolve() if args.labels_out else (out_path.parent / 'labels.txt')

    # Ultralytics forbids optimize=True on CUDA, so force CPU if requested
    export_device = args.device
    if args.optimize:
        export_device = 'cpu'

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Exporting to ONNX: imgsz={args.imgsz}, opset={args.opset}, simplify={args.simplify}, dynamic={args.dynamic}, optimize={args.optimize}, device={export_device}")
    try:
        # Use a deterministic name in the same directory as the desired output to avoid surprises.
        # Ultralytics returns the exported file path.
        exported = model.export(
            format='onnx',
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=bool(args.simplify),
            dynamic=bool(args.dynamic),
            optimize=bool(args.optimize),
            half=False,
            int8=False,
            device=export_device,
            project=str(out_path.parent),
            name=out_path.stem,  # ensures exact filename like model.onnx
            exist_ok=True,
        )
    except Exception as e:
        print(f"ERROR: ONNX export failed: {e}")
        sys.exit(3)

    exported_path = Path(exported).resolve()
    print(f"Ultralytics exported: {exported_path}")

    # Some versions may write with additional suffixes; normalize to --out
    if exported_path != out_path:
        try:
            # Prefer copy + overwrite to be safe across filesystems
            shutil.copyfile(exported_path, out_path)
            print(f"Copied to target path: {out_path}")
        except Exception as e:
            print(f"ERROR: Failed to place ONNX at target path: {out_path}\n{e}")
            sys.exit(4)

    if not out_path.exists():
        print("ERROR: Expected ONNX file not found after export:", out_path)
        sys.exit(5)

    # ---- Write labels.txt next to ONNX ----
    labels: list[str] | None = None
    try:
        if hasattr(model.model, 'names'):
            mnames = model.model.names
            if isinstance(mnames, dict) and mnames:
                # Preserve index order
                try:
                    labels = [mnames[i] for i in range(len(mnames))]
                except Exception:
                    labels = [str(v) for v in mnames.values()]
            elif isinstance(mnames, list) and mnames:
                labels = [str(x) for x in mnames]
    except Exception:
        pass

    if not labels:
        data_yaml = Path(args.data).resolve()
        labels = _load_names_from_yaml(data_yaml)
        if not labels:
            try:
                data_cfg = load_yaml(data_yaml)
                alt = data_cfg.get('classes')
                if isinstance(alt, list) and alt:
                    labels = [str(n).strip() for n in alt]
            except Exception:
                pass

    # Normalize and ensure non-empty
    normalized: list[str] = []
    for i, n in enumerate(labels or []):
        s = str(n).strip()
        if not s:
            s = f'class_{i}'
        normalized.append(s)
    if not normalized:
        normalized = TARGET_CLASSES_30

    ensure_dir(labels_out.parent)
    save_labels_txt(normalized, labels_out)

    # Verify labels file is present and non-empty
    try:
        contents = [ln.strip() for ln in labels_out.read_text(encoding='utf-8').splitlines() if ln.strip()]
        if not contents:
            print(f"ERROR: labels.txt exists but is empty: {labels_out}")
            sys.exit(6)
    except Exception as e:
        print(f"ERROR: Failed to write a valid labels.txt at {labels_out}: {e}")
        sys.exit(6)

    # Basic info print
    try:
        print("torch cuda available:", torch.cuda.is_available())
    except Exception:
        pass

    print("Export complete:")
    print("  ONNX:", out_path)
    print("  Labels:", labels_out)
    # Optional guidance for next steps
    print("Predict:        yolo predict task=detect model=" + str(out_path) + f" imgsz={args.imgsz}")
    print("Validate:       yolo val task=detect model=" + str(out_path) + f" imgsz={args.imgsz} data=" + str(Path(args.data).resolve()))
    print("Visualize:      https://netron.app")


if __name__ == '__main__':
    main()
