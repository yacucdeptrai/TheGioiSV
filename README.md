# TheGioiSV — WildLens (Two-Part Project)

WildLens is a two-part project:
- WildLens-Model: a Python project to train a YOLOv8 object detection model for 30 animal species and export the final ONNX model + labels.
- WildLens-App: a .NET MAUI mobile app that performs real-time camera detection using the exported ONNX model.

This README gives you an end-to-end view: environment setup, data preparation, training/export, ONNX validation, and MAUI app integration.

Quick links:
- WildLens-Model README: `./WildLens-Model/README.md`
- WildLens-App README: `./WildLens-App/README.md`

---

## Repository Structure
- `WildLens-Model/` — Training code, ONNX export, and validation scripts.
- `WildLens-App/` — MAUI mobile app that runs inference on the exported ONNX model.

---

## 1) WildLens-Model (Python / YOLOv8)

Location: `./WildLens-Model`

### 1.1. Environment
- Python 3.10+ is recommended.
- Create and activate a venv:
  ```powershell
  cd .\WildLens-Model
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### 1.2. Data
- Source: Roboflow export for 30 species. Place the dataset (images, labels, `data.yaml`) under `WildLens-Model/data/`.
- Default YAML path: `WildLens-Model/data/data.yaml` (override with `--data` if needed).

### 1.3. Train and Export ONNX
```powershell
# From project root or WildLens-Model folder
cd .\WildLens-Model
.\.venv\Scripts\Activate.ps1  # if not already active

# Train using default data path .\data\data.yaml
python .\train.py --epochs 50 --imgsz 640 --batch 16

# Or specify a custom data.yaml and GPU
python .\train.py --data .\data\data.yaml --epochs 100 --imgsz 640 --batch 32 --device 0 --augment --cache --cos_lr
```
Artifacts:
- `WildLens-Model/exported_models/model.onnx`
- `WildLens-Model/exported_models/labels.txt`

### 1.4. Validate ONNX
```powershell
python .\validate_onnx.py --model .\exported_models\model.onnx --labels .\exported_models\labels.txt --imgsz 640 --providers CPUExecutionProvider
```

For more details, see `WildLens-Model/README.md`.

---

## 2) WildLens-App (.NET MAUI)

Location: `./WildLens-App`

### 2.1. Model Integration
- Copy these files into `WildLens-App/Resources/Assets/`:
  - `WildLens-Model/exported_models/model.onnx`
  - `WildLens-Model/exported_models/labels.txt`
- Ensure their Build Action is `MauiAsset` in `WildLens-App.csproj`.

### 2.2. Build & Run
```powershell
cd .\WildLens-App
# Restore NuGet packages
dotnet restore

# Run on Android emulator/device
dotnet build -t:Run -f net8.0-android

# Or run on iOS (from macOS with required setup)
dotnet build -t:Run -f net8.0-ios
```

For more app-specific details, see `WildLens-App/README.md`.

---

## 3) Troubleshooting
- Training cannot find `data.yaml` → place it at `WildLens-Model/data/data.yaml` or pass `--data`.
- ONNX validation issues → ensure `labels.txt` matches the dataset class order; retrain/export if needed.
- GPU training → use `--device 0` (or `--device cpu`).

---

## 4) License
This repository is provided for educational purposes. Review the licenses of all dependencies (Ultralytics, ONNX Runtime, .NET MAUI, etc.) before distribution.
