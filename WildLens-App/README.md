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

## Build APK (Android)
There are two common outputs for Android:
- APK: for local install/sideloading
- AAB (Android App Bundle): for Play Store upload

### 1) Quick debug APK (for local testing)
```powershell
cd WildLens-App
dotnet publish -f net8.0-android -c Debug -p:AndroidPackageFormat=apk
# Output (example): .\bin\Debug\net8.0-android\publish\com.example.wildlens-Signed.apk
```

### 2) Release APK (unsigned or signed)
Unsigned release APK (useful for CI or manual signing later):
```powershell
cd WildLens-App
dotnet publish -f net8.0-android -c Release -p:AndroidPackageFormat=apk
# Output (examples):
#   .\bin\Release\net8.0-android\publish\com.example.wildlens.apk
#   .\bin\Release\net8.0-android\publish\com.example.wildlens-Signed.apk
```

Signed release APK (provide your keystore info via properties):
```powershell
cd WildLens-App
dotnet publish -f net8.0-android -c Release -p:AndroidPackageFormat=apk `
  -p:AndroidKeyStore=true `
  -p:AndroidSigningKeyStore="C:\path\to\wildlens.keystore" `
  -p:AndroidSigningKeyAlias="wildlens" `
  -p:AndroidSigningKeyPass="<key_password>" `
  -p:AndroidSigningStorePass="<store_password>"
# Output (examples):
#   .\bin\Release\net8.0-android\publish\com.example.wildlens.apk
#   .\bin\Release\net8.0-android\publish\com.example.wildlens-Signed.apk
```

Generate a keystore (one-time) if you don’t have one yet:
```powershell
# Requires JDK's keytool in PATH
keytool -genkey -v -keystore C:\path\to\wildlens.keystore -alias wildlens -keyalg RSA -keysize 2048 -validity 10000
```

### 3) Release AAB (for Play Store)
```powershell
cd WildLens-App
dotnet publish -f net8.0-android -c Release -p:AndroidPackageFormat=aab
# Output (examples):
#   .\bin\Release\net8.0-android\publish\com.example.wildlens.aab
#   .\bin\Release\net8.0-android\publish\com.example.wildlens-Signed.aab
```

Tip: You can also set signing properties in `WildLens-App.csproj` so you don’t pass them on the command line.

## Install APK on a device/emulator
1) Enable developer mode and USB debugging on the Android device (or start an emulator).
2) Ensure the device/emulator is visible:
```powershell
adb devices
```
3) Install the APK (replace path as needed):
```powershell
adb install -r ".\bin\Debug\net8.0-android\publish\com.example.wildlens-Signed.apk"
```
Use the Release path for production builds, e.g.:
```powershell
adb install -r ".\bin\Release\net8.0-android\publish\com.example.wildlens-Signed.apk"
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
