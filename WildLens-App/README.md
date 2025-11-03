# WildLens-App (.NET MAUI)

Cross-platform MAUI app (Android/iOS) that loads an ONNX object detection model and provides a real-time camera view with interactive bounding boxes. Secondary feature: pick an image from gallery for detection. Includes a simple history stored via SQLite.

## Prerequisites
- .NET 8 SDK
- Android/iOS workloads for MAUI installed
- Visual Studio 2022 (Windows) with MAUI tooling, or `dotnet` CLI

## Model & Assets
1. Copy `model.onnx` and `labels.txt` from `../WildLens-Model/exported_models/` into this project's `Resources/Assets/` folder.
2. Ensure `labels.txt` lists class names line-by-line (already included with 30 species as placeholder). The `.csproj` is configured to bundle these as `MauiAsset`.

## Build & Run
```powershell
cd WildLens-App
# Restore NuGet packages
dotnet restore

# Run on Android emulator/device
dotnet build -t:Run -f net8.0-android

# Or run on iOS (from macOS with required setup)
dotnet build -t:Run -f net8.0-ios
```

## Permissions
- Android: Camera and storage permissions are requested by MAUI automatically for CameraView/FilePicker. Ensure your emulator/device has a working camera.
- iOS: Add usage descriptions if needed in `Info.plist` (MAUI templates inject defaults, but you may need to add `NSCameraUsageDescription`).

## Features
- Live camera preview (Camera tab)
- Bounding box overlay with labels + confidence
- Tap a box to select and navigate to Details page
- Pick an image from gallery to run detection
- History tab stores detections (label, confidence, timestamp) via SQLite

## Notes on Real-time Inference
- The code includes an `OnnxInferenceService` that can process RGBA frames. To enable continuous detection on live camera frames, wire up frame callbacks from the CameraView (platform capabilities differ). As a starting point, throttle to ~5 FPS to maintain UI responsiveness.
- Performance tips:
  - Prefer YOLOv8n (nano) sized at 640 or 416 input
  - Consider using half-precision or platform accelerators where supported

## Project Structure
- App.xaml, AppShell.xaml — navigation shell
- Pages/
  - CameraPage — camera preview + overlay; image picker for testing
  - DetailsPage — shows selected detection details
  - HistoryPage — shows stored detections
- ViewModels/ — MVVM bindings for pages
- Services/
  - OnnxInferenceService — loads ONNX and runs inference
  - HistoryService — SQLite wrapper
- Models/ — POCOs for detections and history records
- Resources/Assets/ — model.onnx and labels.txt live here

## Troubleshooting
- If you see "ONNX session not initialized", ensure `Resources/Assets/model.onnx` exists.
- If labels don’t align with detections, verify `labels.txt` order matches training.
- For iOS codesigning/deployment, follow MAUI platform setup docs.
