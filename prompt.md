### AI Agent Prompt: Create an Animal Detection App (Two-Part Project)

**Role:** You are an expert AI/ML engineer and .NET developer, skilled in creating modular and maintainable projects.

**Goal:** Your objective is to complete a two-part project. First, create a self-contained **Python** project to train a **YOLOv8** animal **object detection** model. Second, build a cross-platform **C# .NET MAUI** application whose primary feature is a **real-time camera detector** that consumes the trained model.

---

### Part 1: Model Training Project (`WildLens-Model`)

This project is solely for data preparation, model training, and exporting the final `.onnx` file.

#### **1. Project Structure (`WildLens-Model`)**

Create a directory named `WildLens-Model` containing:
* `data/`: This folder will hold all training and validation datasets (images and labels).
* `scripts/`: Contains the Python scripts for training, validation, and exporting the model.
* `.venv/`: A Python virtual environment for all dependencies.
* `exported_models/`: The final, converted `.onnx` model will be saved here.
* `requirements.txt`: A file listing all Python dependencies.

#### **2. Technical Specifications**
* **Model:** **YOLOv8n (Nano)** for its balance of speed and accuracy.
* **Task:** **Object Detection**. The model must output bounding box coordinates, confidence scores, and class IDs.
* **Data Source:** The training and validation dataset for the 30 species must be sourced, managed, and exported from **Roboflow**.
* **Target Classes (30 species):** The model must be trained to detect the following 30 animal species accurately:
    1.  Dog
    2.  Cat
    3.  Horse
    4.  Cow
    5.  Sheep
    6.  Pig
    7.  Chicken
    8.  Duck
    9.  Bird
    10. Elephant
    11. Lion
    12. Tiger
    13. Bear
    14. Monkey
    15. Deer
    16. Fox
    17. Wolf
    18. Rabbit
    19. Squirrel
    20. Giraffe
    21. Zebra
    22. Kangaroo
    23. Panda
    24. Koala
    25. Raccoon
    26. Penguin
    27. Dolphin
    28. Whale
    29. Turtle
    30. Frog
* **Training Environment:** Python, using the virtual environment at `./.venv`.
* **Core Libraries:**
    * `ultralytics`: For the YOLOv8 framework.
    * `roboflow`: (Optional, but recommended) for dataset downloading.
    * `onnx`: For working with the ONNX format.
    * `onnxruntime`: For validating the exported ONNX model.

#### **3. Workflow**
1.  **Setup:** Initialize the project directory and the Python virtual environment.
2.  **Data:** Source and download the dataset for the 30 target classes from **Roboflow**. Place the exported dataset (including images, labels, and the Roboflow-generated `data.yaml` file) into the `data/` folder.
3.  **Scripting:** Write a Python script in the `scripts/` folder to:
    * Load the pre-trained YOLOv8n model.
    * Fine-tune the **detection** model on the custom dataset located in `data/` (using the `data.yaml` from Roboflow).
    * Export the final, best-performing model to the **ONNX (`.onnx`)** format and a corresponding `labels.txt` file.
4.  **Execution:** Run the training script. The final `model.onnx` and `labels.txt` files must be saved in the `exported_models/` directory.

---

### Part 2: .NET MAUI Application Project (`WildLens-App`)

This is the user-facing mobile application. Its core purpose is to provide an interactive, real-time camera detection experience.

#### **1. Project Structure (`WildLens-App`)**
* Create a standard .NET MAUI project named `WildLens-App`.
* Configure the project to include a `Resources/Assets/` directory.

#### **2. Core Features & Functionality**
* **Live Real-Time Camera Feed:** This is the primary feature. The app must open directly to a camera view (using CameraView) that **continuously processes the live video stream for object detection**.
* **Interactive On-Screen Bounding Boxes:** As animals are detected, the app must draw responsive bounding boxes and labels (species name + confidence score) directly over them in real-time (using a GraphicsView).
* **Multi-Detection Selection:** If multiple animals are detected in a single frame (or a static image), the user must be able to **tap on a specific bounding box** to select it.
* **Image Upload:** As a secondary feature, allow users to pick an image from their gallery (`FilePicker`) for detection. The same interactive bounding box selection logic must apply.
* **Details & History:** Tapping a selected bounding box should navigate the user to a "Details Page" providing more information about that *specific* animal's species. The app should also include a history of past *individual* detections (using `SQLite-net-pcl`).

#### **3. Technical Specifications**
* **Platform:** **.NET MAUI** (for both iOS & Android).
* **Programming Language:** **C#**.
* **Architecture:** **MVVM (Model-View-ViewModel)**.
* **Core NuGet Packages:**
    * `Microsoft.ML.OnnxRuntime`: To load and run the ONNX model.
    * `Microsoft.Maui.Media.Camera`: (Or a community camera library) to get the camera stream.
    * `Microsoft.Maui.Storage.FilePicker`: For file picking.
    * `SkiaSharp.Views.Maui.Controls`: Commonly used for high-performance bounding box drawing and hit-testing (for tapping).

#### **4. Workflow**
1.  **Setup:** Create the .NET MAUI project.
2.  **Model Integration:**
    * **Manually copy** the `model.onnx` **and `labels.txt`** files from the `WildLens-Model/exported_models/` directory into the `WildLens-App/Resources/Assets/` directory.
    * Set the Build Action for these files to `MauiAsset` in the `.csproj` file so they are bundled with the app.
3.  **Development:**
    * Build the UI, prioritizing the camera page with its overlay.
    * Use `Microsoft.ML.OnnxRuntime` (InferenceSession) to load the model and label file from assets.
    * Run inference on the continuous stream of camera frames.
    * Implement the logic to draw bounding boxes based on the model's output onto a `GraphicsView`.
    * Add tap gesture recognition to the `GraphicsView` to detect which bounding box (if any) the user has selected.
4.  **Testing:** Test the app on emulators and physical iOS and Android devices, focusing on detection accuracy, real-time performance, and the responsiveness of tapping on individual bounding boxes.

---

### Final Deliverables

* **Two separate, complete project folders:**
    1.  `WildLens-Model` (The Python Project)
    2.  `WildLens-App` (The C# .NET MAUI Project)
* Each project must have its own `README.md` file with specific setup and usage instructions.