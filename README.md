# TheGioiSV — WildLens (Web + Backend + Model)

>Website URL: https://the-gioi-sv-seven.vercel.app

Unified repository for the WildLens web experience:
- Frontend: Next.js app (React 19) under `Wildlens-Web/Frontend`
- Backend: FastAPI service under `Wildlens-Web/Backend`
- Model: YOLOv11 training/export utilities under `WildLens-Model`

This README explains how to set up, run, and open the website on Windows using the provided helper script or manual commands. It also shows how to bootstrap a fresh Next.js app via `npx create-next-app@latest` if you want to recreate the frontend.

---

## Repository structure
- `Wildlens-Web/`
  - `Backend/` — FastAPI + ONNX Runtime inference API
  - `Frontend/` — Next.js web client
- `WildLens-Model/` — Training, export to ONNX, and validation scripts
- `scripts/` — helper scripts (start/stop, etc.), including `deploy-all.ps1`
- `requirements.txt` — unified Python dependencies for backend/model

Note about generated folders/files:
- During training, Ultralytics may create folders such as `WildLens-Model/scripts/train2` and `WildLens-Model/runs/detect/train*`. These contain training artifacts (plots, logs, `weights/best.pt`). They are safe to keep or delete and are not part of the source code.

## Prerequisites
- Windows PowerShell 5.1+
- Python 3.10 or 3.11 on PATH
- Node.js 18+ (20+ recommended) which includes `npm`
- Optional: NVIDIA GPU with CUDA drivers if you plan to use GPU acceleration

GPU/CPU note for inference dependencies:
- By default we use `onnxruntime-gpu` in `requirements.txt`. For CPU-only machines, replace it with `onnxruntime`.
- Ultralytics may pull a CPU-only Torch. For CUDA-enabled Torch, see the note in `WildLens-Model/README.md`.

---

## Quick start (recommended)

Use the helper script to start the backend and frontend. Make sure you have installed Python dependencies at least once (see Manual setup below).

```powershell
# From repo root
./scripts/deploy-all.ps1
```

What it does:
- Starts FastAPI on `http://127.0.0.1:8000` and Next.js on `http://localhost:3000`
- Sets `NEXT_PUBLIC_API_URL` for the frontend process
- Writes PID files so you can stop/restart later
- Optionally opens the website automatically with `-OpenBrowser`

Other useful commands:
- Stop processes:
  ```powershell
  ./scripts/deploy-all.ps1 -Command stop
  ```
- Restart processes:
  ```powershell
  ./scripts/deploy-all.ps1 -Command restart
  ```
- Status (shows PIDs):
  ```powershell
  ./scripts/deploy-all.ps1 -Command status
  ```

PID files are stored under `./scripts/logs/` as `backend.pid` and `frontend.pid`.

Website URL:
- Frontend: `http://localhost:3000`
- Backend docs (Swagger): `http://127.0.0.1:8000/docs`

---

## Manual setup (advanced)

If you prefer to run parts manually or you need to install dependencies the first time.

### Backend (FastAPI)
```powershell
# From repo root
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r ./requirements.txt

# Run API
cd ./Wildlens-Web/Backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open: `http://127.0.0.1:8000/docs`

### Frontend (Next.js)
```powershell
cd ./Wildlens-Web/Frontend
npm install

# Dev
$Env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev

# Open website
Start-Process http://localhost:3000
```

---

## Bootstrap a fresh Next.js app (optional)

You already have a frontend in `Wildlens-Web/Frontend`. If you want to recreate or start a new one, here’s the official bootstrap command:

```powershell
npx create-next-app@latest frontend
```

This will create a new folder `frontend/` with a Next.js starter. You can then move it under `Wildlens-Web/` or adapt `scripts/deploy-all.ps1` paths if you change the folder name.

---

## Troubleshooting
- First run errors (missing packages): activate your venv and run `pip install -r requirements.txt` in the repo root.
- Locked old virtualenv (Windows): close any running Python/IDE processes, then remove `Wildlens-Web/Backend/.venv` if it still exists.
- Port in use: change `-BackendPort` (script parameter) or stop the conflicting app. Frontend uses port 3000 by default.
- Frontend can’t reach backend: ensure `NEXT_PUBLIC_API_URL` matches your backend host/port.
- PID files: see `./scripts/logs/backend.pid` and `./scripts/logs/frontend.pid`. If a PID file exists but the process is gone, run `-Command status` and then `-Command start`.

---

## Links
- Web (overview, API, tips): `./Wildlens-Web/README.md`
- Backend details: `./Wildlens-Web/Backend/README.md`
- Frontend details: `./Wildlens-Web/Frontend/README.md`
- Model training/export: `./WildLens-Model/README.md`

---

## License
This repository is provided for educational purposes. Review third‑party licenses (Ultralytics, ONNX Runtime, Next.js/React, etc.) before distribution.
