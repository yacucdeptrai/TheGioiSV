# WildLens Web Backend — FastAPI + ONNX Runtime

FastAPI service that runs YOLOv8 ONNX inference and returns detections enriched with species metadata from `species.json`.

- Parent overview and quick start: `../../README.md`
- Web overview and API notes: `../README.md`

---

## Prerequisites
- Windows PowerShell 5.1+ (or any shell; commands below use PowerShell)
- Python 3.10/3.11 on PATH
- Optional: NVIDIA GPU with proper CUDA drivers if using GPU providers

Python dependencies are defined in the repository root `requirements.txt` and shared with the model subproject.

---

## Run (recommended via root helper)
```powershell
# From repository root
./scripts/deploy-all.ps1
```
This starts both backend and frontend. Backend will be at `http://127.0.0.1:8000`.

---

## Run (manual)
```powershell
# From repository root
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r ./requirements.txt

cd ./Wildlens-Web/Backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open docs: `http://127.0.0.1:8000/docs`

---

## Configuration
- Model path/provider: configure in `main.py` as needed.
- CORS: ensure the frontend origin `http://localhost:3000` is allowed.
- Species metadata: edit `species.json` to add fields like `vi_name`, `scientific_name`, `class`, `diet`, `habitat`, `lifespan`, `conservation_status`, `note`.

---

## API
- `POST /detect`
  - Body: multipart form field `file` (image/jpeg or image/png)
  - Returns: list of detections with bounding boxes, labels, confidence, and optional `details` taken from `species.json`.

Example response:
```
{
  "detections": [
    {
      "box": [100, 80, 260, 220],
      "label": "fox",
      "confidence": 0.92,
      "details": {
        "vi_name": "Cáo",
        "scientific_name": "Vulpes vulpes",
        "class": "Mammalia",
        "diet": "Omnivore",
        "habitat": "Rừng ôn đới",
        "lifespan": "3–6 năm",
        "conservation_status": "Least Concern",
        "note": "Hoạt động về đêm."
      }
    }
  ]
}
```

---

## Troubleshooting
- No module named 'uvicorn' → Activate venv and run `pip install -r requirements.txt` in repo root.
- ONNX provider issues → Verify installed `onnxruntime-gpu` or `onnxruntime` matches your environment.
- Large image errors → Add basic size checks or downscale before inference.
- CORS blocked → Confirm FastAPI CORS config allows the frontend origin.

---

## Links
- Web overview: `../README.md`
- Frontend: `../Frontend/README.md`
- Root guide: `../../README.md`
