import fastapi
import uvicorn
import onnxruntime
import cv2
import numpy as np
import json
import io
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

# ==============================================================================
# BƯỚC 3: LOGIC AI CHO YOLOv8 (PHẦN LÕI)
# ==============================================================================

# THAY THẾ TOÀN BỘ HÀM NÀY

def preprocess_image(image_bytes: bytes):
    """
    Tiền xử lý ảnh đầu vào.
    YOLOv8 yêu cầu ảnh 320x320 (dựa theo lỗi của bạn).
    """
    # 1. Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_shape = original_image.shape # (height, width, channels)
    
    # 2. Letterbox (Resize và thêm đệm)
    # ===== THAY ĐỔI Ở ĐÂY =====
    input_height, input_width = 320, 320 # Sửa 640 -> 320
    # ==========================
    
    # Tính tỉ lệ và kích thước mới
    ratio = min(input_width / original_shape[1], input_height / original_shape[0])
    new_shape = (int(original_shape[1] * ratio), int(original_shape[0] * ratio))
    
    # Resize
    resized_image = cv2.resize(original_image, new_shape, interpolation=cv2.INTER_LINEAR)
    
    # Tính toán viền
    delta_w = input_width - new_shape[0]
    delta_h = input_height - new_shape[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Thêm viền (màu 114, 114, 114 là màu xám)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    # 3. Chuẩn hóa và Chuyển đổi
    # (HWC -> CHW) và (BGR -> RGB)
    image_tensor = padded_image.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    image_tensor = np.ascontiguousarray(image_tensor)
    
    # Chuẩn hóa (0-255 -> 0.0-1.0)
    image_tensor = image_tensor.astype(np.float32) / 255.0
    
    # Thêm batch dimension (1, 3, 320, 320)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    
    # Trả về tensor và thông tin để scale lại
    return image_tensor, original_shape, new_shape, (top, left)


# THAY THẾ TOÀN BỘ HÀM NÀY:

def postprocess_output(outputs, original_shape, new_shape, pad, conf_threshold=0.25, nms_threshold=0.45):
    """
    Hậu xử lý đầu ra từ YOLOv8.
    Bao gồm lọc confidence, Non-Max Suppression (NMS) và scale lại bounding box.
    """
    output_data = outputs[0][0] # Đầu ra thường có shape (1, 84, 8400) -> (84, 8400)
    
    # Chuyển (84, 8400) -> (8400, 84)
    output_data = output_data.T 
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Tách box (4) và class scores (80 cho COCO, 30 cho bạn)
    # Lấy 30 nhãn của bạn
    num_classes = len(LABELS) 
    
    for row in output_data:
        # Lấy box [cx, cy, w, h]
        box = row[:4]
        # Lấy confidence của 30 classes
        class_scores = row[4:4 + num_classes] 
        
        # Tìm class có score cao nhất
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        # === CHÚ Ý DÒNG NÀY ===
        # Bây giờ nó sẽ dùng 'conf_threshold' được truyền vào (ví dụ 0.1)
        if confidence > conf_threshold: 
            # Chuyển box [cx, cy, w, h] -> [x1, y1, x2, y2]
            cx, cy, w, h = box
            x1 = int((cx - w / 2))
            y1 = int((cy - h / 2))
            x2 = int((cx + w / 2))
            y2 = int((cy + h / 2))
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            
    # Áp dụng Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Scale box về ảnh gốc
    pad_top, pad_left = pad
    orig_h, orig_w = original_shape[:2]
    new_h, new_w = new_shape[1], new_shape[0] # (w, h) -> (h, w)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            
            # 1. Scale về ảnh đã resize (640x640) nhưng chưa padding
            x1 = (x1 - pad_left)
            y1 = (y1 - pad_top)
            x2 = (x2 - pad_left)
            y2 = (y2 - pad_top)
            
            # 2. Scale về ảnh gốc
            ratio_w = orig_w / new_w
            ratio_h = orig_h / new_h
            
            x1 = int(x1 * ratio_w)
            y1 = int(y1 * ratio_h)
            x2 = int(x2 * ratio_w)
            y2 = int(y2 * ratio_h)

            # Đảm bảo box nằm trong ảnh
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)

            detections.append({
                "class_id": class_ids[i],
                "box": [x1, y1, x2, y2],
                "confidence": confidences[i]
            })
            
    return detections

# ==============================================================================
# BƯỚC 2: FASTAPI SERVER
# ==============================================================================

app = fastapi.FastAPI()

# --- Cấu hình CORS ---
# Cho phép Next.js (localhost:3000) gọi API này (localhost:8000)
origins = [
    "http://localhost:3000",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tải tài nguyên (Model, Labels, Species) khi server khởi động ---
SESSION = None
SPECIES_DATA = {}
LABELS = []

@app.on_event("startup")
def load_resources():
    global SESSION, SPECIES_DATA, LABELS
    
    print("Đang tải tài nguyên...")
    # 1. Tải mô hình ONNX
    SESSION = onnxruntime.InferenceSession("model.onnx")
    
    # 2. Tải file species.json
    with open("species.json", "r", encoding="utf-8") as f:
        SPECIES_DATA = json.load(f)
        
    # 3. Tải file nhãn (labels.txt)
    with open("labels.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Xử lý dòng đầu tiên đặc biệt "Dog"
    first_line_clean = lines[0].split(']')[-1].strip()
    LABELS.append(first_line_clean)
    
    # Xử lý các dòng còn lại
    for line in lines[1:]:
        LABELS.append(line.strip())
        
    print(f"Đã tải {len(LABELS)} nhãn.")
    print(f"Đã tải thông tin {len(SPECIES_DATA)} loài.")
    print("Server sẵn sàng!")


# --- API Endpoint chính ---

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    
    # 1. Tiền xử lý (Bước 3)
    input_tensor, orig_shape, new_shape, pad = preprocess_image(image_bytes)
    
    # 2. Chạy Inference
    input_name = SESSION.get_inputs()[0].name
    outputs = SESSION.run(None, {input_name: input_tensor})
    
    # 3. Hậu xử lý (Bước 3)
    detections = postprocess_output(outputs, orig_shape, new_shape, pad, conf_threshold=0.1)
    
    # 4. Tra cứu thông tin chi tiết
    results_with_info = []
    for det in detections:
        class_id = det["class_id"]
        label = LABELS[class_id] # Lấy tên nhãn (ví dụ: "Dog")
        
        # Lấy thông tin chi tiết từ JSON
        if label in SPECIES_DATA:
            info = SPECIES_DATA[label]
            results_with_info.append({
                "box": det["box"],
                "label": label,
                "confidence": det["confidence"],
                "details": info # Thêm toàn bộ thông tin
            })
            
    return {"detections": results_with_info}

@app.get("/")
def read_root():
    return {"Hello": "Đây là Animal Detector API"}

# --- Lệnh chạy (gõ vào terminal): uvicorn main:app --reload --host 0.0.0.0 --port 8000 ---
# === THÊM VÀO CUỐI FILE main.py ===
if __name__ == "__main__":
    # Lấy port từ biến môi trường của Render, nếu không có thì dùng 8000
    port = int(os.environ.get("PORT", 8000))
    # Thêm 'import os' lên đầu file nhé
    uvicorn.run(app, host="0.0.0.0", port=port)