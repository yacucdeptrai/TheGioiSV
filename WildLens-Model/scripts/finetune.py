import argparse
from pathlib import Path
import shutil
import yaml
import torch

from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_labels_txt(labels, out_path: Path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for name in labels:
            f.write(str(name).strip() + "\n")


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
    return []


def auto_device(user_device: str | None) -> str | int | None:
    if user_device:
        ud = str(user_device).strip().lower()
        if ud in {"cuda", "gpu"}:
            return 0 if torch.cuda.is_available() else "cpu"
        if ud in {"cpu", "mps"}:
            return ud
        return user_device
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='WildLens: Fine-tune or continue training a YOLO model')
    # Data / weights
    parser.add_argument('--data', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'data.yaml'),
                        help='Path to dataset data.yaml')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to base weights .pt to fine-tune from (e.g., best.pt or yolov11n.pt)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Optional: path to an existing run dir or checkpoint (last.pt) to resume exactly')

    # Training knobs
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train in this session')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default=None, help='Device id or type: 0, 0,1, cpu, mps, cuda')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.12)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--cos_lr', action='store_true')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--half', action='store_true', help='Use mixed precision if supported')
    parser.add_argument('--augment', action='store_true', help='Enable stronger augmentation')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--freeze', type=int, default=0,
                        help='Number of backbone layers to freeze (0 = none). See Ultralytics docs for mapping.')
    parser.add_argument('--project', type=str, default=str(Path(__file__).resolve().parent.parent / 'runs'),
                        help='Base project directory for runs (when not resuming)')
    parser.add_argument('--name', type=str, default='finetune', help='Run name (when not resuming)')

    # Export
    parser.add_argument('--export_dir', type=str, default=str(Path(__file__).resolve().parent / 'exported_models'))
    parser.add_argument('--export', action='store_true', help='Export ONNX + labels after fine-tune')

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml not found at {data_yaml}')

    export_dir = Path(args.export_dir).resolve()
    ensure_dir(export_dir)

    dev = auto_device(args.device)

    # Build model
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found: {weights_path}')
    model = YOLO(str(weights_path))

    # Prepare common train params
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        cos_lr=args.cos_lr,
        patience=args.patience,
        half=args.half,
        augment=args.augment,
        cache=args.cache,
        workers=args.workers,
        freeze=args.freeze,
    )

    # Two modes:
    # 1) Resume exact training: --resume points to run dir or last.pt; Ultralytics handles state.
    # 2) Fine-tune from given weights: start a new run with provided hyperparams.
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f'Resume target not found: {resume_path}')
        # If a directory, Ultralytics expects project/name & resume=True. If a file, pass resume=str(file)
        if resume_path.is_dir():
            # Expect structure like runs/detect/train or custom; set project to parent, name to dir name
            project = str(resume_path.parent)
            name = resume_path.name
            model.train(project=project, name=name, resume=True, **train_kwargs)
        else:
            # Checkpoint file case (e.g., last.pt)
            model.train(resume=str(resume_path), **train_kwargs)
    else:
        model.train(project=str(Path(args.project)), name=args.name, **train_kwargs)

    # Locate best checkpoint from the last training call
    last_run = model.trainer.save_dir if hasattr(model, 'trainer') else None
    best_pt = None
    if last_run:
        run_dir = Path(last_run)
        cand = run_dir / 'weights' / 'best.pt'
        if cand.exists():
            best_pt = cand

    # Export ONNX + labels
    if args.export:
        if not best_pt:
            # Fallback: use given weights
            best_pt = weights_path
        best_model = YOLO(str(best_pt))
        onnx_out = Path(export_dir) / 'model.onnx'
        export_device = 'cpu'
        try:
            exported = best_model.export(format='onnx', dynamic=False, simplify=True, opset=12,
                                         half=False, int8=False, device=export_device,
                                         project=str(export_dir), name='model')
            exported_path = Path(exported).resolve() if exported else None
        except Exception as e:
            raise RuntimeError(f'ONNX export failed: {e}')

        # Normalize to export_dir/model.onnx
        try:
            if exported_path and exported_path.exists() and exported_path != onnx_out:
                shutil.copyfile(exported_path, onnx_out)
        except Exception:
            pass
        if not onnx_out.exists():
            # search any .onnx in export_dir
            candidates = list(Path(export_dir).glob('*.onnx'))
            if candidates:
                try:
                    shutil.copyfile(candidates[0], onnx_out)
                except Exception:
                    pass
        if not onnx_out.exists():
            raise RuntimeError('Failed to export ONNX model (no .onnx found).')

        # Write labels.txt
        names = _load_names_from_yaml(data_yaml)
        if names:
            save_labels_txt(names, Path(export_dir) / 'labels.txt')

        print('Export complete:')
        print(f'  ONNX:   {onnx_out}')
        if names:
            print(f'  Labels: {Path(export_dir) / "labels.txt"}')


if __name__ == '__main__':
    main()
