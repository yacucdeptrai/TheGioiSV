# WildLens Web Frontend — Next.js (React 19)

Next.js application that provides the WildLens web UI. It talks to the FastAPI backend via the environment variable `NEXT_PUBLIC_API_URL`.

- Parent overview and quick start: `../../README.md`
- Web overview and API notes: `../README.md`
- Backend service: `../Backend/README.md`

---

## Prerequisites
- Node.js 18+ (20+ recommended)
- npm 9+
- Backend running at `http://127.0.0.1:8000` (default; can be changed)

---

## Run (recommended via root helper)
```powershell
# From repository root
./scripts/deploy-all.ps1 -OpenBrowser
```
This starts both backend and frontend with `NEXT_PUBLIC_API_URL` set automatically.

---

## Run (manual)
```powershell
cd ./Wildlens-Web/Frontend
npm install

# Dev (PowerShell)
$Env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```
Open: `http://localhost:3000`

Production build:
```powershell
npm run build
npm start  # serves the production build
```

---

## Environment variables
- `NEXT_PUBLIC_API_URL` (required): base URL of the FastAPI backend, e.g. `http://127.0.0.1:8000`.
  - Note: Because it starts with `NEXT_PUBLIC_`, it is exposed to the browser at build/runtime as needed.

---

## Features
- Drag & drop upload and click-to-select
- Canvas overlay for bounding boxes and labels
- Rich species details when available from backend (`species.json`), including Vietnamese name, scientific name, class, diet, habitat, lifespan, conservation status, and notes
- Accessible controls, keyboard navigation, responsive layout

---

## Troubleshooting
- Network errors: verify backend is running and `NEXT_PUBLIC_API_URL` is set correctly
- CORS errors: ensure backend allows `http://localhost:3000`
- Port conflicts: stop other apps using port 3000 or change the port via `npm run dev -- -p 3001`

---

## Links
- Backend: `../Backend/README.md`
- Web overview: `../README.md`
- Root guide: `../../README.md`
