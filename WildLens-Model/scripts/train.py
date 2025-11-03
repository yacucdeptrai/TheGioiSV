import argparse
import os
from pathlib import Path
import yaml

from ultralytics import YOLO

TARGET_CLASSES_30 = [
    'Dog','Cat','Horse','Cow','Sheep','Pig','Chicken','Duck','Bird','Elephant',
    'Lion','Tiger','Bear','Monkey','Deer','Fox','Wolf','Rabbit','Squirrel','Giraffe',
    'Zebra','Kangaroo','Panda','Koala','Raccoon','Penguin','Dolphin','Whale','Turtle','Frog'
]

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_labels_txt(labels, out_path: Path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for name in labels:
            f.write(str(name).strip() + "\n")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8n on custom dataset and export ONNX + labels')
    parser.add_argument('--data', type=str, required=True, help='Path to Roboflow data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default=None, help='cuda, 0, 0,1 or cpu')
    parser.add_argument('--export_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'exported_models'))
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml not found at {data_yaml}')

    export_dir = Path(args.export_dir).resolve()
    ensure_dirs(export_dir)

    # Load base YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        plots=True
    )

    # Pick best weights
    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    if not best_weights.exists():
        raise FileNotFoundError(f'best.pt not found under {best_weights}')

    # Reload best model and export ONNX
    best_model = YOLO(str(best_weights))
    onnx_out = export_dir / 'model.onnx'
    best_model.export(format='onnx', imgsz=args.imgsz, opset=12, simplify=True, dynamic=True, optimize=True, half=False, int8=False, 
                      device=args.device, project=str(export_dir), name='', exist_ok=True)

    # The export API may save to export_dir/model.onnx; ensure it exists and move/rename if necessary
    if not onnx_out.exists():
        # Try to find any .onnx in export_dir
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

    print(f'Export complete:')
    print(f'  ONNX:   {onnx_out}')
    print(f'  Labels: {labels_txt}')

if __name__ == '__main__':
    main()
