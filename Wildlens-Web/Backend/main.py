import fastapi
import uvicorn
import onnxruntime
import cv2
import numpy as np
import json
import io
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Deque, Dict, Any
import os
import uuid
from datetime import datetime, timedelta, timezone
from collections import deque
import base64

# ============================================================================== 
# BƯỚC 3: LOGIC AI CHO YOLOv8 (PHẦN LÕI)
# ==============================================================================

# Lưu ý: các biến MODEL_INPUT_W/H và MODEL_LAYOUT sẽ được set khi load model
MODEL_INPUT_W = 640
MODEL_INPUT_H = 640
# layout: "NCHW" (default) hoặc "NHWC"
MODEL_LAYOUT = "NCHW"

def _parse_forced_size_for_env(val: str):
    """
    BACKEND_FORCE_IMG_SIZE có thể là:
      - "640" -> dùng cho cả W và H (vuông)
      - "640x480" -> dùng width=640 height=480 (hoặc "480x640" tùy ý)
    Trả về tuple (w:int, h:int) hoặc None nếu không parse được.
    """
    if not val:
        return None
    try:
        s = val.strip().lower()
        if "x" in s:
            parts = s.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])
        elif s.isdigit():
            v = int(s)
            return v, v
    except Exception:
        pass
    return None

def infer_model_input(session: onnxruntime.InferenceSession):
    """
    Infer input width/height and layout from an ONNX session.
    Returns (w:int, h:int, layout:str)
    layout is "NCHW" or "NHWC".
    """
    # Default fallback
    w, h = 640, 640
    layout = "NCHW"
    try:
        inp = session.get_inputs()[0]
        shape = getattr(inp, "shape", None)  # often like [1, 3, 640, 640] or [None, 3, None, None]
        # helper to coerce digits or numpy ints to int
        def to_int(x):
            try:
                if isinstance(x, (int, np.integer)):
                    return int(x)
                if isinstance(x, str) and x.isdigit():
                    return int(x)
            except Exception:
                return None
            return None

        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
            # Try NCHW: [N, C, H, W]
            c_nchw = to_int(shape[1])
            h_nchw = to_int(shape[2])
            w_nchw = to_int(shape[3])
            # Try NHWC: [N, H, W, C]
            c_nhwc = to_int(shape[3])
            h_nhwc = to_int(shape[1])
            w_nhwc = to_int(shape[2])

            # Determine layout by which axis has C==3
            if c_nchw == 3:
                layout = "NCHW"
                if h_nchw and w_nchw:
                    h, w = h_nchw, w_nchw
            elif c_nhwc == 3:
                layout = "NHWC"
                if h_nhwc and w_nhwc:
                    h, w = h_nhwc, w_nhwc
            else:
                # Fallback heuristics: if either h/w present use them
                if h_nchw and w_nchw:
                    layout = "NCHW"
                    h, w = h_nchw, w_nchw
                elif h_nhwc and w_nhwc:
                    layout = "NHWC"
                    h, w = h_nhwc, w_nhwc
                else:
                    # keep default 640x640
                    pass
        else:
            # shape not helpful, leave defaults
            pass

    except Exception as e:
        print(f"WARN: infer_model_input() failed: {e}")

    return w, h, layout


def preprocess_image(image_bytes: bytes):
    """
    Tiền xử lý ảnh đầu vào.
    Tự động resize và pad ảnh về kích thước yêu cầu của model (MODEL_INPUT_W x MODEL_INPUT_H).
    Hỗ trợ cả layout NCHW và NHWC.
    Trả về: input_tensor phù hợp để đưa vào ONNX, cùng original_shape, new_shape, pad=(top,left)
    """
    global MODEL_INPUT_W, MODEL_INPUT_H, MODEL_LAYOUT

    # 1. Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError("Unable to decode input image")
    orig_h, orig_w = original_image.shape[:2]
    original_shape = original_image.shape  # (h, w, c)

    # 2. Lấy kích thước mục tiêu (cho phép override bằng ENV)
    forced = _parse_forced_size_for_env(os.environ.get("BACKEND_FORCE_IMG_SIZE", "") or "")
    if forced:
        target_w, target_h = forced
    else:
        target_w, target_h = MODEL_INPUT_W, MODEL_INPUT_H

    # Tính tỉ lệ và kích thước mới (maintain aspect + letterbox)
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    new_shape = (new_w, new_h)  # (w, h) để tiện tính scale sau

    # Resize
    resized_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tính padding để đạt target size
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Thêm viền (màu 114, 114, 114)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])

    # Chuẩn hóa: convert BGR -> RGB
    img_rgb = padded_image[:, :, ::-1]

    # Chuyển và chuẩn hoá theo layout
    if MODEL_LAYOUT == "NCHW":
        # HWC -> CHW
        img_chw = img_rgb.transpose((2, 0, 1))
        img_chw = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
        input_tensor = np.expand_dims(img_chw, axis=0)  # (1, C, H, W)
    else:
        # NHWC: (1, H, W, C)
        img_nhwc = np.ascontiguousarray(img_rgb, dtype=np.float32) / 255.0
        input_tensor = np.expand_dims(img_nhwc, axis=0)

    return input_tensor, original_shape, new_shape, (top, left)


def postprocess_output(outputs, original_shape, new_shape, pad, conf_threshold=0.25, nms_threshold=0.45):
    """
    Hậu xử lý đầu ra từ YOLOv8.
    Lưu ý: logic này giả định output dạng (1, num_attrs, num_boxes) hoặc (1, num_boxes, num_attrs).
    Nếu model của bạn trả ra khác (ví dụ: boxes + scores riêng biệt), cần sửa lại.
    """
    # Heuristic: chọn tensor outputs[0], cố gắng transpose nếu cần
    out = outputs[0]
    # Nếu shape (1, attrs, boxes) -> squeeze first dim
    if isinstance(out, np.ndarray) and out.ndim == 3 and out.shape[0] == 1:
        out_proc = out[0]  # (attrs, boxes)
        # transpose to (boxes, attrs)
        out_proc = out_proc.T
    else:
        # try to make it (boxes, attrs)
        if isinstance(out, np.ndarray) and out.ndim == 2:
            out_proc = out
        else:
            out_proc = np.array(out)
            if out_proc.ndim == 3 and out_proc.shape[0] == 1:
                out_proc = out_proc[0].T
            elif out_proc.ndim == 2:
                pass
            else:
                # fallback
                out_proc = out_proc.reshape(-1, out_proc.shape[-1])

    output_data = out_proc  # shape (N, attrs)

    boxes = []
    confidences = []
    class_ids = []

    num_classes = len(LABELS)

    for row in output_data:
        # First 4 are box (cx, cy, w, h) or (x, y, x2, y2) depending on model
        box = row[:4]
        class_scores = row[4:4 + num_classes]
        if len(class_scores) == 0:
            # If model provides objectness and then class probs differently, skip
            continue
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence > conf_threshold:
            cx, cy, w, h = box
            x1 = int((cx - w / 2))
            y1 = int((cy - h / 2))
            x2 = int((cx + w / 2))
            y2 = int((cy + h / 2))

            boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    pad_top, pad_left = pad
    orig_h, orig_w = original_shape[:2]
    new_w, new_h = new_shape  # new_shape was (w, h)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            # Remove padding
            x1 = x1 - pad_left
            y1 = y1 - pad_top
            x2 = x2 - pad_left
            y2 = y2 - pad_top

            # Scale back to original image
            ratio_w = orig_w / new_w
            ratio_h = orig_h / new_h

            x1 = int(x1 * ratio_w)
            y1 = int(y1 * ratio_h)
            x2 = int(x2 * ratio_w)
            y2 = int(y2 * ratio_h)

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

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION = None
SPECIES_DATA = {}
LABELS = []

# Short-term history config
HISTORY_TTL_MINUTES = int(os.environ.get("HISTORY_TTL_MINUTES", "30"))
HISTORY_MAX_ITEMS = int(os.environ.get("HISTORY_MAX_ITEMS", "500"))

class HistoryStore:
    def __init__(self, ttl_minutes: int = 30, max_items: int = 500):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_items = max_items
        self.records: Dict[str, Dict[str, Any]] = {}
        self.order: Deque[str] = deque()

    def _cleanup(self):
        now = datetime.now(timezone.utc)
        expired_ids = []
        for rid, rec in list(self.records.items()):
            ts = rec.get("ts_dt")
            if ts and now - ts > self.ttl:
                expired_ids.append(rid)
        for rid in expired_ids:
            self.records.pop(rid, None)
        if len(self.records) > self.max_items:
            ids_by_time = sorted(self.records.items(), key=lambda kv: kv[1].get("ts_dt", now))
            to_remove = len(self.records) - self.max_items
            for i in range(to_remove):
                self.records.pop(ids_by_time[i][0], None)

    def add(self, record: Dict[str, Any]) -> str:
        self._cleanup()
        rid = str(uuid.uuid4())
        record = dict(record)
        record["id"] = rid
        ts_dt = datetime.now(timezone.utc)
        record["ts_dt"] = ts_dt
        record["ts_iso"] = ts_dt.isoformat()
        self.records[rid] = record
        self.order.append(rid)
        while len(self.order) > self.max_items:
            old_id = self.order.popleft()
            self.records.pop(old_id, None)
        return rid

    def list(self) -> List[Dict[str, Any]]:
        self._cleanup()
        items = sorted(self.records.values(), key=lambda r: r.get("ts_dt"), reverse=True)
        out = []
        for r in items:
            labels = [d.get("label") for d in r.get("detections", [])]
            out.append({
                "id": r["id"],
                "ts": r.get("ts_iso"),
                "labels": labels,
                "count": len(labels),
                "thumb_b64": r.get("thumb_b64"),
            })
        return out

    def get(self, rid: str) -> Dict[str, Any] | None:
        self._cleanup()
        rec = self.records.get(rid)
        if not rec:
            return None
        return {
            "id": rec["id"],
            "ts": rec.get("ts_iso"),
            "detections": rec.get("detections", []),
            "image_b64": rec.get("image_b64"),
        }

HISTORY = HistoryStore(ttl_minutes=HISTORY_TTL_MINUTES, max_items=HISTORY_MAX_ITEMS)

@app.on_event("startup")
def load_resources():
    global SESSION, SPECIES_DATA, LABELS, MODEL_INPUT_W, MODEL_INPUT_H, MODEL_LAYOUT

    print("Đang tải tài nguyên...")
    # 1. Tải mô hình ONNX
    model_path = os.environ.get("BACKEND_ONNX_PATH", "model.onnx")
    SESSION = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Suy ra kích thước đầu vào & layout từ mô hình
    try:
        inferred_w, inferred_h, inferred_layout = infer_model_input(SESSION)
        # Allow env override (single value or WxH)
        forced = _parse_forced_size_for_env(os.environ.get("BACKEND_FORCE_IMG_SIZE", "") or "")
        if forced:
            MODEL_INPUT_W, MODEL_INPUT_H = forced
        else:
            MODEL_INPUT_W, MODEL_INPUT_H = inferred_w, inferred_h
        MODEL_LAYOUT = inferred_layout
        print(f"INFERRED ONNX input size: {MODEL_INPUT_W}x{MODEL_INPUT_H}, layout: {MODEL_LAYOUT}")
    except Exception as e:
        print(f"WARN: Unable to infer model input size, using default {MODEL_INPUT_W}x{MODEL_INPUT_H}. Error: {e}")

    # 2. Tải file species.json
    with open("species.json", "r", encoding="utf-8") as f:
        SPECIES_DATA = json.load(f)

    # 3. Tải file nhãn (labels.txt)
    with open("labels.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    first_line_clean = lines[0].split(']')[-1].strip()
    LABELS.append(first_line_clean)
    for line in lines[1:]:
        LABELS.append(line.strip())

    print(f"Đã tải {len(LABELS)} nhãn.")
    print(f"Đã tải thông tin {len(SPECIES_DATA)} loài.")
    print("Server sẵn sàng!")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor, orig_shape, new_shape, pad = preprocess_image(image_bytes)

    # run inference
    input_name = SESSION.get_inputs()[0].name
    # ensure dtype matches (float32)
    if isinstance(input_tensor, np.ndarray):
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)
    outputs = SESSION.run(None, {input_name: input_tensor})

    detections = postprocess_output(outputs, orig_shape, new_shape, pad, conf_threshold=0.1)

    results_with_info = []
    for det in detections:
        class_id = det["class_id"]
        label = LABELS[class_id] if class_id < len(LABELS) else f"cls{class_id}"
        if label in SPECIES_DATA:
            info = SPECIES_DATA[label]
            results_with_info.append({
                "box": det["box"],
                "label": label,
                "confidence": det["confidence"],
                "details": info
            })

    # Save thumbnails + history (same logic)
    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            max_side = max(h, w)
            scale = 1280.0 / max_side if max_side > 1280 else 1.0
            if scale != 1.0:
                img_disp = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            else:
                img_disp = img
            ok, enc = cv2.imencode('.jpg', img_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            image_b64 = None
            if ok:
                image_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode('ascii')
            th_scale = 280.0 / max(img_disp.shape[0], img_disp.shape[1]) if max(img_disp.shape[:2]) > 280 else 1.0
            thumb = cv2.resize(img_disp, (int(img_disp.shape[1]*th_scale), int(img_disp.shape[0]*th_scale)), interpolation=cv2.INTER_AREA) if th_scale != 1.0 else img_disp
            ok2, enc2 = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            thumb_b64 = None
            if ok2:
                thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(enc2.tobytes()).decode('ascii')
        else:
            image_b64 = None
            thumb_b64 = None
    except Exception:
        image_b64 = None
        thumb_b64 = None

    record_id = HISTORY.add({
        "detections": results_with_info,
        "image_b64": image_b64,
        "thumb_b64": thumb_b64,
        "filename": getattr(file, 'filename', None),
        "size": len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else None,
    })

    return {"detections": results_with_info, "record_id": record_id}

@app.get("/")
def read_root():
    return {"Hello": "Đây là Animal Detector API"}

@app.get("/history")
def list_history():
    return {"items": HISTORY.list(), "ttl_minutes": HISTORY_TTL_MINUTES}

@app.get("/history/{record_id}")
def get_history(record_id: str):
    item = HISTORY.get(record_id)
    if not item:
        raise fastapi.HTTPException(status_code=404, detail="Record not found or expired")
    return item

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
