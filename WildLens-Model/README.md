# WildLens-Model

Python project to fine-tune YOLOv8n for 30 animal species and export an ONNX model plus labels.txt for consumption by the WildLens-App (.NET MAUI).

## Structure
- data/ — place your Roboflow-exported dataset here (images, labels, data.yaml)
- scripts/
  - train.py — fine-tunes YOLOv8n, exports ONNX + labels
  - validate_onnx.py — quick ONNX sanity check with onnxruntime
- exported_models/ — final artifacts will be saved here (model.onnx, labels.txt)
- requirements.txt — Python dependencies

## Setup
1. Create a virtual environment and install deps:
   - Windows PowerShell:
     ```powershell
     cd WildLens-Model
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -U pip
     pip install -r requirements.txt
     ```
2. Download dataset from Roboflow (30 target classes) and export in YOLO format. Unzip into `data/` so that `data/data.yaml` exists and points to train/val images and labels.

## Train and Export
```powershell
# From project root/WildLens-Model with venv activated
python scripts/train.py --data .\data\data.yaml --epochs 50 --imgsz 640 --batch 16
```
Artifacts are saved to `exported_models/model.onnx` and `exported_models/labels.txt`.

## Validate ONNX
```powershell
python scripts/validate_onnx.py --model .\exported_models\model.onnx --labels .\exported_models\labels.txt
```

## Target Classes (30)
Dog, Cat, Horse, Cow, Sheep, Pig, Chicken, Duck, Bird, Elephant, Lion, Tiger, Bear, Monkey, Deer, Fox, Wolf, Rabbit, Squirrel, Giraffe, Zebra, Kangaroo, Panda, Koala, Raccoon, Penguin, Dolphin, Whale, Turtle, Frog.
