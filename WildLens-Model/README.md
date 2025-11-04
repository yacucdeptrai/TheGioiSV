# WildLens-Model — Training, Export, and Validation (YOLOv8)

This subproject contains everything needed to train a YOLOv8 model for 30 animal species and export it to ONNX for use in the MAUI app.

- If you’re looking for the overall project documentation, see the root README at `../README.md`.
- If you want to integrate the exported model into the app, see `../WildLens-App/README.md`.

---

## 1) Structure
- `data/` — dataset location (images, labels, Roboflow-style `data.yaml`).
  - 30 class folders are pre-created as placeholders; actual training reads from `data.yaml`.
- `exported_models/` — outputs: `model.onnx` and `labels.txt`.
- `train.py` — improved training and ONNX export script.
- `validate_onnx.py` — enhanced ONNX validation script.
- `requirements.txt` — Python dependencies.

## 2) Environment
- Python 3.10+ is recommended.
- Create and activate a virtual environment on Windows PowerShell:
  ```powershell
  cd .\WildLens-Model
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## 3) Dataset
- Place your Roboflow export under `WildLens-Model/data/` so that a `data.yaml` exists at `WildLens-Model/data/data.yaml` (default expected by `train.py`).
- You can override the path with `--data`.
- Target classes (30): Dog, Cat, Horse, Cow, Sheep, Pig, Chicken, Duck, Bird, Elephant, Lion, Tiger, Bear, Monkey, Deer, Fox, Wolf, Rabbit, Squirrel, Giraffe, Zebra, Kangaroo, Panda, Koala, Raccoon, Penguin, Dolphin, Whale, Turtle, Frog.

## 4) Train and Export ONNX
```powershell
# From this folder
cd .\WildLens-Model
.\.venv\Scripts\Activate.ps1  # if not already active

# Train using default data path .\data\data.yaml
python .\train.py --epochs 50 --imgsz 640 --batch 16

# Or specify a custom data.yaml and GPU
python .\train.py --data .\data\data.yaml --epochs 100 --imgsz 640 --batch 32 --device 0 --augment --cache --cos_lr
```
Outputs:
- `exported_models/model.onnx`
- `exported_models/labels.txt`

## 5) Validate ONNX
```powershell
python .\validate_onnx.py --model .\exported_models\model.onnx --labels .\exported_models\labels.txt --imgsz 640 --providers CPUExecutionProvider
```
What it does:
- Verifies ONNX integrity with `onnx.checker`.
- Prints model inputs/outputs and runs dummy inferences at 320/480/640 (and your `--imgsz`).
- Ensures returned tensors are non-empty and shaped reasonably.

## 6) Tips & Troubleshooting
- `FileNotFoundError: data.yaml not found` → put it at `WildLens-Model/data/data.yaml` or pass `--data`.
- Output shape warnings during validation can occur across Ultralytics versions; ensure `labels.txt` class count matches your dataset.
- For GPU training, use `--device 0` (or `--device cpu` to force CPU).

## 7) Links
- Overall project guide: `../README.md`
- MAUI app integration: `../WildLens-App/README.md`

## 8) License
This subproject is for educational purposes. Check third‑party licenses (Ultralytics, ONNX Runtime, etc.) before redistribution.
