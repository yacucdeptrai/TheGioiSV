### WildLens-Web

Modern, mobile‑first web client and API for wildlife detection.

Parts:
- Frontend: Next.js (React 19) under `Wildlens-Web/Frontend`
- Backend: FastAPI under `Wildlens-Web/Backend` running a YOLOv8 ONNX model

Tip: From the repo root you can run one command to start everything and open the website:
```powershell
./scripts/deploy-all.ps1 -OpenBrowser
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
   │  ├─ layout.tsx         # App layout
   │  └─ page.module.css    # Styles for home page
   └─ package.json
```

---

### Prerequisites

- Node.js ≥ 18 and npm ≥ 9
- Python ≥ 3.10 (recommended 3.10/3.11)

On Windows, use PowerShell for the commands below.

---

### Start via the root helper (recommended)

```powershell
# From repo root
./scripts/deploy-all.ps1
```

What it does
- Starts backend dev server at `http://127.0.0.1:8000`
- Starts frontend dev server at `http://localhost:3000` with `NEXT_PUBLIC_API_URL`
- Writes PID files to `./scripts/logs/backend.pid` and `./scripts/logs/frontend.pid`

Useful commands
```powershell
./scripts/deploy-all.ps1 -Command status
./scripts/deploy-all.ps1 -Command stop
./scripts/deploy-all.ps1 -Command restart
```

---

### Backend (manual)

```powershell
# From repo root
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r ./requirements.txt   # unified dependencies at repo root

cd ./Wildlens-Web/Backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Notes
- Dependencies come from the repo root `requirements.txt` (shared by backend and model).
- The service loads an ONNX model via ONNX Runtime. Configure model path in `main.py` as needed.
- `species.json` provides label details (VN name, habitat, class, scientific name, etc.) used by the UI.

---

### Frontend (manual)

```powershell
cd Wildlens-Web/Frontend
npm install

# Dev (PowerShell)
$Env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

Open the app at `http://localhost:3000`.

---

### How it works (high level)

- Upload an image (click or drag & drop).
- The frontend sends the image to the FastAPI `/detect` endpoint.
- The backend preprocesses to 320×320, runs YOLOv8 ONNX inference, postprocesses (score filter + NMS), rescales boxes, and returns detections with metadata from `species.json`.
- The frontend draws the image and bounding boxes on a canvas and shows rich species info (VN name, scientific name, class, diet, habitat, lifespan, conservation status, notes when available).

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

---

### Troubleshooting

- No module named 'uvicorn' → Activate venv and run `pip install -r requirements.txt`.
- CORS errors → Ensure FastAPI allows `http://localhost:3000` and the backend is reachable.
- Frontend network error / 404 → Verify `NEXT_PUBLIC_API_URL`; check backend console output.
- ONNX/OpenCV errors → Confirm compatible versions in `requirements.txt`.

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

### Bootstrap a new frontend (optional)

```powershell
npx create-next-app@latest frontend
```

Move it under `Wildlens-Web/` and adjust paths if you change folder names.

---

### License

Add your preferred license here (e.g., MIT). Ensure third‑party model/data licenses are respected.

---

### Acknowledgements

- Built with Next.js, FastAPI, ONNX Runtime, OpenCV.
- YOLO family by Ultralytics and the broader open‑source community.
