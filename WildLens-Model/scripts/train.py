import argparse
import os
from pathlib import Path
import yaml
import torch

from ultralytics import YOLO

TARGET_CLASSES_30 = [
    'Dog','Cat','Horse','Cow','Sheep','Pig','Chicken','Duck','Bird','Elephant',
    'Lion','Tiger','Bear','Monkey','Deer','Fox','Wolf','Rabbit','Squirrel','Giraffe',
    'Zebra','Kangaroo','Panda','Koala','Raccoon','Penguin','Dolphin','Whale','Turtle','Frog'
]


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
        # Handle dict form: {0: 'cls0', 1: 'cls1', ...} or {'0': 'cls0', ...}
        if isinstance(names, dict) and names:
            try:
                items = sorted(((int(k), v) for k, v in names.items()), key=lambda t: t[0])
                return [str(v).strip() for _, v in items]
            except Exception:
                # Fallback: keep insertion order of values
                return [str(v).strip() for v in names.values()]
        if isinstance(names, list) and names:
            return [str(n).strip() for n in names if str(n).strip()]
    except Exception:
        pass
    return TARGET_CLASSES_30


def _build_name2id(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def _detect_species_from_path(label_file: Path, valid: dict[str, int]) -> tuple[str, int] | None:
    p = label_file
    # Walk up a few levels: .../<Species>/<split>/labels/file.txt
    for _ in range(6):
        p = p.parent
        if not p:
            break
        if p.name in valid:
            return p.name, valid[p.name]
    return None


def _remap_label_file(file_path: Path, new_id: int, backup: bool = True) -> int:
    text = file_path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out_lines = []
    changed = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            out_lines.append(ln)
            continue
        parts = s.split()
        try:
            int(parts[0])
        except Exception:
            out_lines.append(ln)
            continue
        parts[0] = str(new_id)
        out_lines.append(' '.join(parts))
        changed += 1
    if backup:
        file_path.with_suffix('.txt.bak').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    file_path.write_text('\n'.join(out_lines) + ('\n' if out_lines else ''), encoding='utf-8')
    return changed


def _scan_unique_ids(root: Path) -> list[int]:
    ids = set()
    for f in root.rglob('labels/*.txt'):
        for line in f.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s:
                continue
            tok = s.split()[0]
            if tok.isdigit():
                ids.add(int(tok))
    return sorted(ids)


def auto_device(user_device: str | None) -> str | int | None:
    """Select device with priority: explicit user → CUDA (0) → MPS → CPU.
    Accepts common synonyms like 'cuda', 'cpu', 'mps', or GPU indices '0', '0,1'.
    """
    if user_device:
        ud = str(user_device).strip().lower()
        # Normalize common synonyms
        if ud in {"cuda", "gpu"}:
            return 0 if torch.cuda.is_available() else "cpu"
        if ud in {"cpu", "mps"}:
            return ud
        # If a comma-separated list or numeric id is provided, just forward to Ultralytics
        return user_device
    # Prefer CUDA if available
    if torch.cuda.is_available():
        return 0
    # Mac MPS
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='WildLens: Train YOLOv8 on custom 30-species dataset and export ONNX + labels')
    parser.add_argument('--data', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'data.yaml'),
                        help='Path to dataset data.yaml (centralized at WildLens-Model/data/data.yaml by default)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default=None, help='Device id or type: 0, 0,1, cpu, mps, cuda')
    parser.add_argument('--export_dir', type=str, default=str(Path(__file__).resolve().parent / 'exported_models'))
    # Advanced knobs
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.003, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.12, help='final OneCycleLR factor')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--cos_lr', action='store_true', help='use cosine LR scheduler')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--half', action='store_true', help='train in mixed precision if supported')
    parser.add_argument('--augment', action='store_true', help='enable stronger data augmentation')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--workers', type=int, default=8, help='number of dataloader workers')
    # Label remapping & safety
    parser.add_argument('--auto-remap', action='store_true', help='Automatically remap label class ids based on species folder names before training')
    parser.add_argument('--remap-no-backup', action='store_true', help='Do not write .bak backups during auto-remap')
    parser.add_argument('--force', action='store_true', help='Force training even if a label sanity check looks wrong (e.g., all ids are 0)')
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml not found at {data_yaml}')

    export_dir = Path(args.export_dir).resolve()
    ensure_dir(export_dir)

    dev = auto_device(args.device)

    # Log device selection and environment
    try:
        if isinstance(dev, (int, str)):
            if dev == 'cpu':
                print('Device selected: CPU')
            elif dev == 'mps':
                print('Device selected: Apple MPS (Metal)')
            else:
                # e.g., 0 or '0,1'
                print(f'Device selected: {dev} (CUDA preferred when available)')
        else:
            print(f'Device selected: {dev}')

        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                current = 0 if isinstance(dev, int) else torch.cuda.current_device()
                name = torch.cuda.get_device_name(current)
                cap = torch.cuda.get_device_capability(current)
                print(f"Using CUDA device: {current} — {name}, capability={cap}")
            except Exception:
                pass
        else:
            # If ORT sees CUDA but PyTorch does not, warn user how to fix
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    print("WARN: ONNX Runtime exposes CUDAExecutionProvider but PyTorch is CPU-only.")
                    print("      Install a CUDA-enabled PyTorch matching your driver:")
                    print("      python .\\WildLens-Model\\scripts\\install_torch_cuda.py --yes")
                    print("      Or manually: pip uninstall -y torch torchvision torchaudio && "
                          "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio")
            except Exception:
                pass
    except Exception:
        pass

    # ----- Label sanity check and optional auto-remap -----
    data_root = Path(__file__).resolve().parent.parent / 'data'
    names = _load_names_from_yaml(data_yaml)
    name2id = _build_name2id(names)

    try:
        unique_ids_before = _scan_unique_ids(data_root)
        print(f"Label sanity check: unique class ids found before remap: {unique_ids_before}")
        if args.auto_remap:
            print('Auto-remap enabled: remapping label files to global ids based on species folder names...')
            files_processed = 0
            lines_changed = 0
            for lbl in data_root.rglob('labels/*.txt'):
                det = _detect_species_from_path(lbl, name2id)
                if not det:
                    continue
                species, gid = det
                changed = _remap_label_file(lbl, gid, backup=not args.remap_no_backup)
                if changed:
                    files_processed += 1
                    lines_changed += changed
            unique_ids_after = _scan_unique_ids(data_root)
            print(f"Auto-remap complete. Files touched: {files_processed}, lines changed: {lines_changed}")
            print(f"Unique class ids after remap: {unique_ids_after}")
        else:
            # If everything is class 0, warn and suggest options
            if unique_ids_before == [0] and not args.force:
                print('ERROR: All labels currently use class id 0. This usually means each species folder kept its local class index.\n'
                      '       Run the remapper before training:')
                print('       python .\\WildLens-Model\\scripts\\remap_labels.py')
                print('       Or re-run train with --auto-remap, or override with --force to proceed anyway.')
                return
    except Exception as e:
        print(f'WARN: Label sanity check failed: {e}')

    # Load base YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Compose training kwargs using Ultralytics API flags
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        patience=args.patience,
        project=str(export_dir.parent),  # train runs will go under WildLens-Model/runs/
        verbose=True,
        plots=True,
        workers=args.workers,
        # optimizer & LR
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        # precision/caching
        amp=args.half,  # use AMP (mixed precision) when available
        cache=args.cache,
    )

    # Optional augmentations (align with Ultralytics names)
    if args.augment:
        train_kwargs.update(dict(
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
            perspective=0.0, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.1
        ))

    # Train
    results = model.train(**train_kwargs)

    # Pick best weights
    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    if not best_weights.exists():
        raise FileNotFoundError(f'best.pt not found under {best_weights}')

    # Reload best model and export ONNX
    best_model = YOLO(str(best_weights))
    onnx_out = export_dir / 'model.onnx'

    # Export with robust params; ultralytics will create file inside export_dir
    # Note: Ultralytics asserts optimize=True is not compatible with CUDA devices.
    # To keep optimized ONNX while training on GPU, we export on CPU.
    export_device = 'cpu'
    if isinstance(dev, str) and dev.lower() == 'cpu' or (isinstance(dev, int) and not torch.cuda.is_available()):
        export_device = 'cpu'
    print(f"Exporting ONNX on device: {export_device} (optimize=True requires CPU)")

    # Perform export and capture the actual path returned by Ultralytics
    try:
        exported = best_model.export(
            format='onnx',
            imgsz=args.imgsz,
            opset=12,
            simplify=True,
            dynamic=True,
            optimize=True,
            half=False,
            int8=False,
            device=export_device,
            project=str(export_dir),
            name='model',  # ensure deterministic filename 'model.onnx' in export_dir
            exist_ok=True,
        )
    except Exception as e:
        raise RuntimeError(f'ONNX export failed: {e}')

    # Normalize the output location to export_dir/model.onnx even if Ultralytics wrote elsewhere
    try:
        exported_path = Path(exported).resolve()
    except Exception:
        exported_path = onnx_out if onnx_out.exists() else None

    if exported_path and exported_path.exists() and exported_path != onnx_out:
        try:
            # Copy to the expected canonical location
            import shutil
            shutil.copyfile(exported_path, onnx_out)
            print(f"Copied exported ONNX from {exported_path} to {onnx_out}")
        except Exception as e:
            print(f"WARN: Failed to copy exported ONNX to canonical path {onnx_out}: {e}")

    # The export API may save to export_dir/model.onnx; ensure it exists and move/rename if necessary
    if not onnx_out.exists():
        # Fallback: try to find any .onnx in export_dir and normalize the name
        onnx_candidates = list(export_dir.glob('*.onnx'))
        if onnx_candidates:
            try:
                # Prefer copy to avoid cross-device rename issues
                import shutil
                shutil.copyfile(onnx_candidates[0], onnx_out)
                print(f"Placed ONNX at {onnx_out} from {onnx_candidates[0]}")
            except Exception as e:
                print(f"WARN: Couldn't place ONNX at {onnx_out} from {onnx_candidates[0]}: {e}")

    if not onnx_out.exists():
        # As a last resort, try to detect common Ultralytics output locations (e.g., runs/*/weights/best.onnx)
        candidates = []
        try:
            # search within scripts/ and runs/
            scripts_dir = Path(__file__).resolve().parent
            model_root = scripts_dir.parent
            candidates += list(scripts_dir.rglob('weights/*.onnx'))
            runs_dir = model_root / 'runs'
            if runs_dir.exists():
                candidates += list(runs_dir.rglob('weights/*.onnx'))
        except Exception:
            pass
        if candidates:
            # Pick most recent
            candidates = sorted({p.resolve() for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                import shutil
                shutil.copyfile(candidates[0], onnx_out)
                print(f"Recovered ONNX to {onnx_out} from {candidates[0]}")
            except Exception:
                pass

    if not onnx_out.exists():
        raise RuntimeError('Failed to export ONNX model (no .onnx found).')

    # Prepare labels.txt, prefer model.names else from data.yaml
    labels = None
    if hasattr(best_model.model, 'names'):
        mnames = best_model.model.names
        if isinstance(mnames, dict):
            labels = [mnames[i] for i in range(len(mnames))]
        elif isinstance(mnames, list):
            labels = mnames
    if not labels:
        # Load from YAML (supports both list and dict forms); fallback to TARGET_CLASSES_30
        labels = _load_names_from_yaml(data_yaml)
        # If YAML didn't have names but had an alternate key
        try:
            if not labels:
                data_cfg = load_yaml(data_yaml)
                alt = data_cfg.get('classes')
                if isinstance(alt, list) and alt:
                    labels = [str(n).strip() for n in alt]
        except Exception:
            pass

    # Normalize, ensure non-empty strings
    normalized_labels: list[str] = []
    for i, n in enumerate(labels or []):
        s = str(n).strip()
        if not s:
            s = f'class_{i}'
        normalized_labels.append(s)
    if not normalized_labels:
        # Last resort fallback
        normalized_labels = TARGET_CLASSES_30

    labels_txt = export_dir / 'labels.txt'
    save_labels_txt(normalized_labels, labels_txt)

    # Verify labels file is present and non-empty
    try:
        contents = [ln.strip() for ln in labels_txt.read_text(encoding='utf-8').splitlines() if ln.strip()]
        if not contents:
            raise RuntimeError('labels.txt exists but is empty')
    except Exception as e:
        raise RuntimeError(f'Failed to write a valid labels.txt at {labels_txt}: {e}')

    print('Export complete:')
    print(f'  ONNX:   {onnx_out}')
    print(f'  Labels: {labels_txt}')
    # Helpful next-step hints
    print(f"Predict:        yolo predict task=detect model={onnx_out} imgsz={args.imgsz}")
    print(f"Validate:       yolo val task=detect model={onnx_out} imgsz={args.imgsz} data={data_yaml}")
    print("Visualize:      https://netron.app")


if __name__ == '__main__':
    main()
