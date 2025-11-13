### WildLens-Web

Modern, mobile‑first web interface and API for wildlife detection.

Parts:
- Frontend: Next.js (React 19) under `Wildlens-Web/Frontend`
- Backend: FastAPI under `Wildlens-Web/Backend` running a YOLOv8 ONNX model

Tip: From the repo root you can run one command to start everything and open the website:
```powershell
./scripts/deploy-all.ps1
```
Frontend: http://localhost:3000
Backend docs: http://127.0.0.1:8000/docs

---

### Project structure

```
Wildlens-Web/
├─ Backend/
│  ├─ main.py               # FastAPI app with ONNXRuntime inference
│  └─ species.json          # Label → metadata mapping used in responses
└─ Frontend/
   ├─ app/
   │  ├─ page.tsx           # Home page with upload + canvas results
   │  ├─ layout.tsx         # Global layout incl. Header/Footer
   │  └─ globals.css        # Design system and global styles
   ├─ components/
   │  ├─ Header.tsx
   │  └─ Footer.tsx
   ├─ public/
   └─ package.json
```

---

### Prerequisites

- Node.js ≥ 18 and npm ≥ 9
- Python ≥ 3.10 (recommended 3.10/3.11)
- Git (optional)

On Windows, use PowerShell for the commands below.

---

### Backend (FastAPI + ONNXRuntime)

Option A — via the root deploy script (recommended)
```powershell
# From repo root
./scripts/deploy-all.ps1
```

Option B — manual (advanced)
```powershell
# From repo root
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r ./requirements.txt   # unified dependencies at repo root

cd ./Wildlens-Web/Backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`.

Notes
- Dependencies come from the repo root `requirements.txt` (shared by backend and model).
- The service loads an ONNX model via ONNX Runtime. Place or configure your model path in `main.py` as needed.
- `species.json` provides label details (VN name, habitat, etc.) used by the UI.

---

### Frontend (Next.js)

1) Install dependencies

```powershell
cd Wildlens-Web/Frontend
npm install
```

2) Configure API URL (optional)

By default, the frontend calls `http://127.0.0.1:8000`. To override, set `NEXT_PUBLIC_API_URL`:

PowerShell (Windows)
```powershell
$Env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"; npm run dev
```

macOS/Linux
```bash
NEXT_PUBLIC_API_URL="http://127.0.0.1:8000" npm run dev
```

3) Start the dev server

```powershell
npm run dev
```

Open the app at `http://localhost:3000`.

---

### How it works (high level)

- Upload an image (click or drag & drop).
- The frontend sends the image to the FastAPI `/detect` endpoint.
- The backend preprocesses to 320×320, runs YOLOv8 ONNX inference, postprocesses (score filter + NMS), rescales boxes, and returns detections with metadata from `species.json`.
- The frontend draws the image and bounding boxes on a `<canvas>`, and lists detection details.

---

### UI/UX highlights

- Mobile‑first layout, responsive grid for results (canvas + info panel)
- Sticky, accessible header with hamburger menu; keyboard and ESC support
- Clear loading/error/empty states; improved contrast and focus styles
- Centralized design tokens (colors, spacing, radii, shadows) in `globals.css`
- Subtle transitions; respects reduced‑motion preferences

---

### Production builds

Frontend
```
cd Wildlens-Web/Frontend
npm run build
npm start   # serves the production build
```

Backend
```
cd Wildlens-Web/Backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

For Docker, you can create simple Dockerfiles for each service and run them with a compose file mapping ports 3000 and 8000.

---

### One-command deploy on Windows (PowerShell)

For a quick local deployment that brings up both the backend (FastAPI) and frontend (Next.js) with sensible defaults, use the provided script:

```powershell
# From the repository root
./scripts/deploy-all.ps1
```

What it does
- Creates a Python virtual environment at repo root `.venv` (unless `-NoVenv`).
- Installs root `requirements.txt` and frontend `npm` dependencies (use `-ReinstallDeps` to force).
- Starts the backend at `http://127.0.0.1:8000` (dev uses `--reload`).
- Sets `NEXT_PUBLIC_API_URL` and starts the frontend at `http://localhost:3000`.
- Waits for readiness and prints PIDs and log file locations under `logs/`.

Common options
```powershell
# Production-like run (no reload, Next.js production build/start):
./scripts/deploy-all.ps1 -Mode prod -BackendHost 0.0.0.0 -BackendPort 8000 -FrontendPort 3000

# Reinstall deps and auto-open browser:
./scripts/deploy-all.ps1 -ReinstallDeps -OpenBrowser

# Stop/restart/status for processes started by the script:
./scripts/deploy-all.ps1 -Command stop
./scripts/deploy-all.ps1 -Command restart
./scripts/deploy-all.ps1 -Command status
```

Notes
- If PowerShell blocks the script, run with `-ExecutionPolicy Bypass` once.
- Logs are stored in `logs/backend.*.log` and `logs/frontend.*.log` at the repo root.
- PIDs are stored in `.wildlens-state/pids.json` for the `-Command` actions to use.

---

### Troubleshooting

- No module named 'uvicorn'
  - Activate venv and run `pip install -r requirements.txt`.
- CORS errors in browser console
  - Ensure FastAPI has CORS enabled for `http://localhost:3000` and that the backend is reachable.
- Frontend shows network error / 404
  - Verify `NEXT_PUBLIC_API_URL` matches your backend URL; check the backend logs.
- ONNX / OpenCV errors
  - Confirm compatible versions in `requirements.txt`. Some ONNX models need specific opsets/providers.

---

### API

`POST /detect`
- Body: multipart form field `file` (image/png, image/jpeg)
- Response example
```
{
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "label": "fox",
      "confidence": 0.92,
      "details": {
        "vi_name": "Cáo",
        "habitat": "Rừng ôn đới",
        "lifespan": "3–6 năm",
        "note": "Hoạt động về đêm."
      }
    }
  ]
}
```

---

### Bootstrap a new frontend (optional)

You already have `Wildlens-Web/Frontend`. If you want to recreate a fresh app:

```powershell
npx create-next-app@latest frontend
```

Then move or adapt it under `Wildlens-Web/` and update `scripts/deploy-all.ps1` if needed.

---

### Development tips

- Keep backend and frontend in separate terminals during development.
- Use browser devtools Network tab to inspect requests to `/detect`.
- Adjust confidence/NMS thresholds in backend postprocessing if needed.

---

### License

Add your preferred license here (e.g., MIT). Ensure third‑party model/data licenses are respected.

---

### Acknowledgements

- Built with Next.js, FastAPI, ONNX Runtime, OpenCV.
- YOLO family by Ultralytics and the broader open‑source community.
