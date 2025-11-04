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


def auto_device(user_device: str | None) -> str | int | None:
    if user_device:
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
    parser.add_argument('--data', type=str, default=str(Path(__file__).resolve().parent / 'data' / 'data.yaml'),
                        help='Path to Roboflow data.yaml (default: ./data/data.yaml)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
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
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml not found at {data_yaml}')

    export_dir = Path(args.export_dir).resolve()
    ensure_dir(export_dir)

    dev = auto_device(args.device)

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
        workers=8,
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
    best_model.export(
        format='onnx',
        imgsz=args.imgsz,
        opset=12,
        simplify=True,
        dynamic=True,
        optimize=True,
        half=False,
        int8=False,
        device=dev,
        project=str(export_dir),
        name='',
        exist_ok=True,
    )

    # The export API may save to export_dir/model.onnx; ensure it exists and move/rename if necessary
    if not onnx_out.exists():
        onnx_candidates = list(export_dir.glob('*.onnx'))
        if onnx_candidates:
            onnx_candidates[0].rename(onnx_out)

    if not onnx_out.exists():
        raise RuntimeError('Failed to export ONNX model')

    # Prepare labels.txt, prefer model.names else from data.yaml
    labels = None
    if hasattr(best_model.model, 'names'):
        mnames = best_model.model.names
        if isinstance(mnames, dict):
            labels = [mnames[i] for i in range(len(mnames))]
        elif isinstance(mnames, list):
            labels = mnames
    if not labels:
        data_cfg = load_yaml(data_yaml)
        labels = data_cfg.get('names') or data_cfg.get('classes') or TARGET_CLASSES_30

    labels_txt = export_dir / 'labels.txt'
    save_labels_txt(labels, labels_txt)

    print('Export complete:')
    print(f'  ONNX:   {onnx_out}')
    print(f'  Labels: {labels_txt}')


if __name__ == '__main__':
    main()
