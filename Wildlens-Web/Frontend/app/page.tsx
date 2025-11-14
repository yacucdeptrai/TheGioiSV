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
    const [lastRecordId, setLastRecordId] = useState<string | null>(null);
    const [imgSrc, setImgSrc] = useState<string | null>(null); // Để lưu ảnh gốc
    const [isDragOver, setIsDragOver] = useState<boolean>(false);
    // Detect mobile device (client-side only)
    const [isMobile, setIsMobile] = useState<boolean | null>(null);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    // Separate hidden inputs for mobile: camera vs gallery
    const cameraInputRef = useRef<HTMLInputElement>(null);
    const galleryInputRef = useRef<HTMLInputElement>(null);

    // Simple device detection with multiple fallbacks, runs only on client
    useEffect(() => {
        const detectMobile = (): boolean => {
            try {
                // Newer Chromium
                // @ts-ignore
                if (navigator.userAgentData && typeof navigator.userAgentData.mobile === 'boolean') {
                    // @ts-ignore
                    return navigator.userAgentData.mobile;
                }
            } catch { /* ignore */ }
            const ua = (typeof navigator !== 'undefined' ? navigator.userAgent || '' : '').toLowerCase();
            const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/;
            const isTouchCapable = typeof window !== 'undefined' && ('ontouchstart' in window || (navigator as any).maxTouchPoints > 0);
            return mobileRegex.test(ua) || isTouchCapable;
        };
        setIsMobile(detectMobile());
    }, []);

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

            const data: ApiResponse & { record_id?: string } = await response.json();
            setDetections(data.detections || []);
            if (data.record_id) {
                setLastRecordId(data.record_id);
            }

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

    // Mobile: trigger native camera via hidden input with capture
    const triggerMobileCamera = () => {
        cameraInputRef.current?.click();
    };

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
                  className={`${styles.dropzone} dropzone ${!isMobile && isDragOver ? 'dragover' : ''}`} 
                  onDrop={!isMobile ? onDrop : undefined}
                  onDragOver={!isMobile ? onDragOver : undefined}
                  onDragLeave={!isMobile ? onDragLeave : undefined}
                  role="region"
                  aria-label={isMobile ? 'Chọn nguồn ảnh' : 'Kéo thả ảnh vào đây hoặc chọn ảnh từ máy'}
                >
                  <div className={styles.dropInner}>
                    {isMobile ? (
                      <>
                        <p><strong>Chọn nguồn ảnh</strong></p>
                        <div className={styles.actionsRow}>
                          <button
                            type="button"
                            className="btn primary"
                            onClick={triggerMobileCamera}
                            aria-label="Chụp bằng camera"
                          >Dùng camera</button>
                          <label htmlFor="gallery-input" className="btn" aria-label="Chọn ảnh từ thư viện">Chọn từ thư viện</label>
                        </div>
                        {/* Hidden input that hints the OS to open Camera; OS may still allow choosing from gallery as fallback */}
                        <input
                          ref={cameraInputRef}
                          id="camera-input"
                          type="file"
                          accept="image/*"
                          capture="environment"
                          onChange={handleFileChange}
                          className="visually-hidden"
                        />
                        {/* Hidden input for picking from gallery explicitly */}
                        <input
                          ref={galleryInputRef}
                          id="gallery-input"
                          type="file"
                          accept="image/*,.jpg,.jpeg,.png,.webp,.avif,.bmp,.gif,.tif,.tiff"
                          onChange={handleFileChange}
                          className="visually-hidden"
                        />
                        <p className="muted" aria-live="polite">Nhấn “Dùng camera” để mở ứng dụng Camera trên thiết bị. Nếu không hỗ trợ, hệ điều hành sẽ mở trình chọn ảnh.</p>
                      </>
                    ) : (
                      <>
                        <p><strong>Kéo & thả ảnh</strong> vào đây hoặc</p>
                        <div className={styles.actionsRow}>
                          <label htmlFor="file-input" className="btn primary" aria-label="Chọn ảnh từ máy">Chọn ảnh</label>
                        </div>
                        <input
                          id="file-input"
                          type="file"
                          accept="image/*,.jpg,.jpeg,.png,.webp,.avif,.bmp,.gif,.tif,.tiff"
                          onChange={handleFileChange}
                          className="visually-hidden"
                        />
                      </>
                    )}
                    <p className="muted" aria-live="polite">Hỗ trợ JPG, PNG, WebP, AVIF, BMP, GIF, TIFF (tùy hỗ trợ trình duyệt).</p>
                  </div>
                </div>

                {isLoading && <div className={styles.banner} role="status">Đang xử lý... Vui lòng chờ.</div>}
                {error && <div className={`${styles.banner} ${styles.error}`} role="alert">{error}</div>}
                {!isLoading && !error && lastRecordId && (
                  <div className={styles.banner} role="status" aria-live="polite">
                    Đã lưu vào Lịch sử (tồn tại 30 phút). 
                    <a href={`/history/${lastRecordId}`} style={{ marginLeft: 8 }}>Xem chi tiết</a> ·
                    <a href={`/history`} style={{ marginLeft: 8 }}>Mở Lịch sử</a>
                  </div>
                )}
            </section>

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