'use client';

import { useState, useRef, useEffect, ChangeEvent, DragEvent } from 'react';
import styles from './page.module.css';

// --- Định nghĩa "kiểu" dữ liệu cho TypeScript ---

// 1. Định nghĩa kiểu cho 1 đối tượng được phát hiện
interface DetectionDetail {
    vi_name?: string;
    habitat?: string;
    lifespan?: string;
    note?: string;
    // New/extended fields from Backend species.json
    class?: string;
    scientific_name?: string;
    diet?: string;
    conservation_status?: string;
}

interface DetectionResult {
    box: [number, number, number, number]; // [x1, y1, x2, y2]
    label: string;
    confidence: number;
    details?: DetectionDetail;
}

// 2. Định nghĩa kiểu cho API response
interface ApiResponse {
    detections: DetectionResult[];
}

// --- Component chính ---
export default function Home() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [detections, setDetections] = useState<DetectionResult[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [imgSrc, setImgSrc] = useState<string | null>(null); // Để lưu ảnh gốc
    const [isDragOver, setIsDragOver] = useState<boolean>(false);
    const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);
    const [cameraError, setCameraError] = useState<string | null>(null);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);

    // Hàm xử lý khi người dùng chọn file
    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]; // Lấy file đầu tiên
        if (file) {
            setSelectedFile(file);
            setDetections([]); // Xóa kết quả cũ
            setError(null);
            
            const reader = new FileReader();
            reader.onload = (e: ProgressEvent<FileReader>) => {
                // e.target.result là một string (data URL)
                setImgSrc(e.target?.result as string); 
            };
            reader.readAsDataURL(file);
            
            // Tự động gọi nhận diện khi chọn ảnh
            handleDetection(file);
        }
    };

    // Drag and Drop handlers
    const onDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) {
            const fakeEvent = { target: { files: [file] } } as unknown as ChangeEvent<HTMLInputElement>;
            handleFileChange(fakeEvent);
        }
    };
    const onDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragOver(true);
    };
    const onDragLeave = () => setIsDragOver(false);

    // Hàm gửi ảnh đi để nhận diện
    const handleDetection = async (fileToUpload: File) => {
        if (!fileToUpload) return;

        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", fileToUpload);

        try {
            // Gọi đến Backend FastAPI (đang chạy trên port 8000)
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

            const response = await fetch(`${apiUrl}/detect`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Lỗi từ server: ${response.statusText}`);
            }

            const data: ApiResponse = await response.json();
            setDetections(data.detections || []);

        } catch (err) {
            if (err instanceof Error) {
                setError(err.message || "Có lỗi xảy ra khi gọi API. Đảm bảo Backend đang chạy.");
            } else {
                setError("Có lỗi không xác định xảy ra.");
            }
        } finally {
            setIsLoading(false);
        }
    };

    // CAMERA: mở camera và xin quyền
    const openCamera = async () => {
        setCameraError(null);
        setError(null);
        try {
            if (!('mediaDevices' in navigator) || !navigator.mediaDevices?.getUserMedia) {
                throw new Error('Trình duyệt không hỗ trợ camera (getUserMedia).');
            }
            // Ưu tiên camera sau (environment) nếu có
            const constraints: MediaStreamConstraints = {
                video: { facingMode: { ideal: 'environment' } },
                audio: false,
            };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            mediaStreamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }
            setIsCameraOpen(true);
        } catch (e) {
            console.error(e);
            setCameraError('Không thể truy cập camera. Vui lòng cấp quyền hoặc kiểm tra thiết bị.');
        }
    };

    const closeCamera = () => {
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach((t) => t.stop());
            mediaStreamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setIsCameraOpen(false);
    };

    // Chụp ảnh từ video và gửi nhận diện
    const captureFromCamera = async () => {
        if (!videoRef.current) return;
        const video = videoRef.current;

        // Tạo canvas tạm để chụp frame
        const temp = document.createElement('canvas');
        const w = video.videoWidth;
        const h = video.videoHeight;
        temp.width = w;
        temp.height = h;
        const tctx = temp.getContext('2d');
        if (!tctx) return;
        tctx.drawImage(video, 0, 0, w, h);

        // Hiển thị trước lên UI
        const dataUrl = temp.toDataURL('image/jpeg', 0.95);
        setImgSrc(dataUrl);
        setDetections([]);

        // Chuyển thành blob/file để tận dụng pipeline có sẵn
        const blob = await new Promise<Blob | null>((resolve) => temp.toBlob((b) => resolve(b), 'image/jpeg', 0.95));
        if (blob) {
            const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
            setSelectedFile(file);
            await handleDetection(file);
        } else {
            setError('Không thể chụp ảnh từ camera.');
        }
        // Đóng camera sau khi chụp để tiết kiệm pin
        closeCamera();
    };

    // Đảm bảo dọn tài nguyên camera khi rời trang/unmount
    useEffect(() => {
        return () => {
            if (mediaStreamRef.current) {
                mediaStreamRef.current.getTracks().forEach((t) => t.stop());
                mediaStreamRef.current = null;
            }
        };
    }, []);

    // Hàm vẽ kết quả lên canvas
    useEffect(() => {
        // Đảm bảo canvas đã sẵn sàng
        if (!canvasRef.current) return;
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) return; // Không thể lấy context
        
        if (imgSrc) {
            const img = new Image();
            img.onload = () => {
                // Set kích thước canvas bằng kích thước ảnh gốc
                canvas.width = img.width;
                canvas.height = img.height;
                
                // 1. Vẽ ảnh gốc lên
                try {
                    ctx.drawImage(img, 0, 0);
                } catch (e) {
                    console.error('Không thể vẽ ảnh. Trình duyệt có thể không hỗ trợ định dạng này.', e);
                    setError('Trình duyệt không hỗ trợ định dạng ảnh này. Hãy thử JPG/PNG/WebP/AVIF.');
                    return;
                }

                // 2. Vẽ các hộp Bounding Box
                detections.forEach((det: DetectionResult) => {
                    const { box, label, confidence } = det;
                    
                    // box = [x1, y1, x2, y2]
                    ctx.strokeStyle = '#00FF00'; // Màu xanh lá
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
                    
                    // Vẽ nền cho text
                    ctx.fillStyle = '#00FF00';
                    const displayLabel = det.details?.vi_name ? `${det.details.vi_name} | ${label}` : label;
                    const text = `${displayLabel} (${(confidence * 100).toFixed(0)}%)`;
                    ctx.font = '18px Arial';
                    const textMetrics = ctx.measureText(text);
                    const textWidth = textMetrics.width;
                    
                    const textX = box[0];
                    const textY = box[1] > 20 ? box[1] - 20 : box[1];

                    ctx.fillRect(textX, textY, textWidth + 4, 20);
                    
                    // Vẽ text
                    ctx.fillStyle = '#000000'; // Màu chữ đen
                    ctx.fillText(
                        text, 
                        textX + 2, 
                        textY + 15
                    );
                });
            };
            img.onerror = () => {
                setError('Không thể tải ảnh. Định dạng có thể không được hỗ trợ trên trình duyệt này.');
            };
            img.src = imgSrc;
        } else {
            // Xóa canvas nếu không có ảnh
             ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

    }, [imgSrc, detections]); // Chạy lại khi 2 giá trị này thay đổi

    return (
        <div className="container">
            <section id="upload" className={styles.section} aria-labelledby="upload-title">
                <h1 id="upload-title" className={styles.title}>WildLens — Nhận diện Động vật</h1>
                <p className={styles.subtitle}>Chụp ảnh hoặc tải ảnh lên để hệ thống AI nhận diện loài động vật trong hình.</p>

                <div 
                  className={`${styles.dropzone} dropzone ${isDragOver ? 'dragover' : ''}`} 
                  onDrop={onDrop} 
                  onDragOver={onDragOver} 
                  onDragLeave={onDragLeave}
                  role="button"
                  aria-label="Kéo thả ảnh vào đây để tải lên"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      const input = document.getElementById('file-input') as HTMLInputElement | null;
                      input?.click();
                    }
                  }}
                >
                  <div className={styles.dropInner}>
                    <p><strong>Kéo & thả ảnh</strong> vào đây hoặc</p>
                    <div className={styles.actionsRow}>
                      <label htmlFor="file-input" className="btn primary" aria-label="Chọn ảnh từ máy">Chọn ảnh</label>
                      <button type="button" className="btn" onClick={openCamera} aria-label="Mở camera để chụp ảnh">Dùng camera</button>
                    </div>
                    <input
                      id="file-input"
                      type="file"
                      accept="image/*,.jpg,.jpeg,.png,.webp,.avif,.bmp,.gif,.tif,.tiff"
                      // gợi ý dùng camera sau trên di động hỗ trợ
                      capture="environment"
                      onChange={handleFileChange}
                      className="visually-hidden"
                    />
                    <p className="muted" aria-live="polite">Hỗ trợ JPG, PNG, WebP, AVIF, BMP, GIF, TIFF (tùy hỗ trợ trình duyệt).</p>
                  </div>
                </div>

                {isLoading && <div className={styles.banner} role="status">Đang xử lý... Vui lòng chờ.</div>}
                {(error || cameraError) && <div className={`${styles.banner} ${styles.error}`} role="alert">{error || cameraError}</div>}
            </section>

            {isCameraOpen && (
              <section id="camera" className={styles.section} aria-labelledby="camera-title">
                <h2 id="camera-title" className={styles.sectionTitle}>Camera</h2>
                <div className={styles.cameraBox}>
                  <video ref={videoRef} className={styles.video} playsInline muted />
                  <div className={styles.cameraControls}>
                    <button className="btn primary" onClick={captureFromCamera} aria-label="Chụp ảnh">Chụp ảnh</button>
                    <button className="btn" onClick={closeCamera} aria-label="Đóng camera">Đóng</button>
                  </div>
                  <p className="muted">Nếu không thấy camera, hãy cấp quyền truy cập camera cho trình duyệt.</p>
                </div>
              </section>
            )}

            <section id="results" className={styles.section} aria-labelledby="results-title">
                <h2 id="results-title" className={styles.sectionTitle}>Kết quả</h2>

                <div className={styles.resultsGrid}>
                    <figure className={styles.canvasCard} aria-labelledby="figure-caption">
                        <canvas 
                          ref={canvasRef} 
                          className={styles.canvas}
                          aria-label={imgSrc ? 'Ảnh đã tải lên với khung nhận diện' : 'Chưa có ảnh để hiển thị'}
                          role="img"
                        />
                        <figcaption id="figure-caption" className="visually-hidden">
                          Ảnh gốc và các khung bao của các đối tượng được nhận diện.
                        </figcaption>
                    </figure>

                    <div className={styles.infoBox}>
                        {!imgSrc && (
                          <p className="muted">Chưa có ảnh. Hãy tải ảnh ở phần trên.</p>
                        )}
                        {detections.length === 0 && imgSrc && !isLoading && (
                            <div className={styles.empty}>
                              <p>Không phát hiện thấy động vật nào trong ảnh này.</p>
                            </div>
                        )}
                        {detections.map((det: DetectionResult, index: number) => (
                            <article key={index} className={styles.infoCard} aria-labelledby={`det-${index}-title`}>
                                <h3 id={`det-${index}-title`}>
                                  {det.details?.vi_name ?? det.label}
                                  <span className={styles.label}> ({det.label})</span>
                                  <span className={styles.conf}> {(det.confidence * 100).toFixed(0)}%</span>
                                </h3>
                                <ul className={styles.detailsList}>
                                    {det.details?.scientific_name && (
                                      <li><strong>Tên khoa học:</strong> <em>{det.details.scientific_name}</em></li>
                                    )}
                                    {det.details?.class && (
                                      <li><strong>Ngành/Lớp:</strong> {det.details.class}</li>
                                    )}
                                    {det.details?.diet && (
                                      <li><strong>Chế độ ăn:</strong> {det.details.diet}</li>
                                    )}
                                    {det.details?.habitat && (
                                      <li><strong>Nơi sống:</strong> {det.details.habitat}</li>
                                    )}
                                    {det.details?.lifespan && (
                                      <li><strong>Tuổi thọ:</strong> {det.details.lifespan}</li>
                                    )}
                                    {det.details?.conservation_status && (
                                      <li><strong>Tình trạng bảo tồn:</strong> {det.details.conservation_status}</li>
                                    )}
                                    {det.details?.note && (
                                      <li><strong>Ghi chú:</strong> {det.details.note}</li>
                                    )}
                                </ul>
                            </article>
                        ))}
                    </div>
                </div>
            </section>

            <section id="about" className={styles.section} aria-labelledby="about-title">
                <h2 id="about-title" className={styles.sectionTitle}>Về WildLens</h2>
                <p className="muted">WildLens sử dụng mô hình YOLO qua FastAPI backend để nhận diện các loài động vật trong ảnh. Trải nghiệm được tối ưu cho di động và hỗ trợ truy cập bằng bàn phím.</p>
            </section>
        </div>
    );
}