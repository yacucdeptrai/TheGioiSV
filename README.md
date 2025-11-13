# TheGioiSV — WildLens (Web + Backend + Model)

Unified repository for the WildLens web experience:
- Frontend: Next.js app (React 19) under `Wildlens-Web/Frontend`
- Backend: FastAPI service under `Wildlens-Web/Backend`
- Model: YOLOv8 training/export utilities under `WildLens-Model`

This README explains how to set up, run, and open the website using a single command on Windows. It also shows how you could bootstrap a fresh Next.js app via `npx create-next-app@latest` if you ever want to recreate the frontend.

---

## Repository structure
- `Wildlens-Web/`
  - `Backend/` — FastAPI + ONNX Runtime inference API
  - `Frontend/` — Next.js web client
- `WildLens-Model/` — Training, export to ONNX, and validation scripts
- `scripts/` — helper scripts, including `deploy-all.ps1`
- `requirements.txt` — unified Python dependencies for backend/model
- `logs/` — runtime logs written by the deploy script

Note on generated folders/files:
- During training, Ultralytics may create folders such as `WildLens-Model/scripts/train2` and `WildLens-Model/runs/detect/train*`. These contain training artifacts (plots, logs, `weights/best.pt`). They are safe to keep or delete and are not part of the source code.

---

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

Use the helper script to install dependencies, start backend and frontend, and open the website.

```powershell
# From repo root
./scripts/deploy-all.ps1
```

What it does:
- Creates a single virtual environment at repo root (`.venv`)
- Installs Python deps from the root `requirements.txt`
- Installs Node deps for the Next.js frontend
- Starts FastAPI on `http://127.0.0.1:8000` and Next.js on `http://localhost:3000`
- Saves PIDs so you can stop/restart later
- Optionally opens the website automatically

Other useful commands:
- Start in production mode (no auto-reload):
  ```powershell
  ./scripts/deploy-all.ps1 -Mode prod -BackendHost 0.0.0.0 -BackendPort 8000 -FrontendPort 3000
  ```
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

Logs are written to `./logs/backend.*.log` and `./logs/frontend.*.log`.

Website URL:
- Frontend: `http://localhost:3000`
- Backend docs (Swagger): `http://127.0.0.1:8000/docs`

---

## Manual setup (advanced)

If you prefer to run parts manually.

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

This will create a new folder `frontend/` with a Next.js starter. You can then move it under `Wildlens-Web/` or adapt the `scripts/deploy-all.ps1` script to point at the new path.

---

## Troubleshooting
- Locked old virtualenv (Windows): close any running Python/IDE processes, then remove `Wildlens-Web/Backend/.venv` if it still exists.
- Port in use: change `-BackendPort` / `-FrontendPort` or stop the conflicting app.
- Frontend can’t reach backend: ensure `NEXT_PUBLIC_API_URL` matches your backend host/port.
- Check logs: see `./logs/` for `backend.out.log`, `backend.err.log`, `frontend.out.log`, `frontend.err.log`.

---

## Links
- Backend and Frontend guide: `./Wildlens-Web/README.md`
- Model training/export: `./WildLens-Model/README.md`

---

## License
This repository is provided for educational purposes. Review third‑party licenses (Ultralytics, ONNX Runtime, Next.js/React, etc.) before distribution.
