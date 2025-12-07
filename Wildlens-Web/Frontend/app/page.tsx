'use client';

import { useState, useRef, useEffect, ChangeEvent, DragEvent } from 'react';
import styles from './page.module.css';

// --- Type definitions ---
interface DetectionDetail {
    vi_name?: string;
    habitat?: string;
    lifespan?: string;
    note?: string;
    class?: string;
    scientific_name?: string;
    diet?: string;
    conservation_status?: string;
}

interface DetectionResult {
    box: [number, number, number, number];
    label: string;
    confidence: number;
    details?: DetectionDetail;
}

interface ApiResponse {
    detections: DetectionResult[];
}

// Upload icon component
const UploadIcon = ({ className }: { className?: string }) => (
    <svg 
        className={className} 
        viewBox="0 0 24 24" 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
    >
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
);

// Camera icon component
const CameraIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
        <circle cx="12" cy="13" r="4"/>
    </svg>
);

// Gallery icon component
const GalleryIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
    </svg>
);

// Loading shimmer component
const LoadingShimmer = () => (
    <div className={styles.loading}>
        <div className={styles.loadingSpinner} />
        <span>Đang xử lý ảnh... Vui lòng chờ</span>
    </div>
);

export default function Home() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [detections, setDetections] = useState<DetectionResult[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [lastRecordId, setLastRecordId] = useState<string | null>(null);
    const [imgSrc, setImgSrc] = useState<string | null>(null);
    const [isDragOver, setIsDragOver] = useState<boolean>(false);
    const [isMobile, setIsMobile] = useState<boolean | null>(null);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const cameraInputRef = useRef<HTMLInputElement>(null);
    const galleryInputRef = useRef<HTMLInputElement>(null);

    // Device detection
    useEffect(() => {
        const detectMobile = (): boolean => {
            try {
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

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setDetections([]);
            setError(null);
            
            const reader = new FileReader();
            reader.onload = (e: ProgressEvent<FileReader>) => {
                setImgSrc(e.target?.result as string); 
            };
            reader.readAsDataURL(file);
            
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

    // Resize image to max 320px dimension before sending to backend
    const resizeImageTo320 = (file: File): Promise<Blob> => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            img.onload = () => {
                const maxSize = 320;
                let { width, height } = img;

                // Calculate new dimensions maintaining aspect ratio
                if (width > height) {
                    if (width > maxSize) {
                        height = Math.round((height * maxSize) / width);
                        width = maxSize;
                    }
                } else {
                    if (height > maxSize) {
                        width = Math.round((width * maxSize) / height);
                        height = maxSize;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                
                if (ctx) {
                    ctx.drawImage(img, 0, 0, width, height);
                    canvas.toBlob(
                        (blob) => {
                            if (blob) {
                                resolve(blob);
                            } else {
                                reject(new Error('Failed to create blob'));
                            }
                        },
                        'image/jpeg',
                        0.85
                    );
                } else {
                    reject(new Error('Failed to get canvas context'));
                }
            };

            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = URL.createObjectURL(file);
        });
    };

    const handleDetection = async (fileToUpload: File) => {
        if (!fileToUpload) return;

        setIsLoading(true);
        setError(null);

        try {
            // Resize image to 320px before sending
            const resizedBlob = await resizeImageTo320(fileToUpload);
            
            const formData = new FormData();
            formData.append("file", resizedBlob, fileToUpload.name);

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

    const triggerMobileCamera = () => {
        cameraInputRef.current?.click();
    };

    // Draw results on canvas
    useEffect(() => {
        if (!canvasRef.current) return;
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) return;
        
        if (imgSrc) {
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                
                try {
                    ctx.drawImage(img, 0, 0);
                } catch (e) {
                    console.error('Cannot draw image', e);
                    setError('Trình duyệt không hỗ trợ định dạng ảnh này.');
                    return;
                }

                // Draw bounding boxes
                detections.forEach((det: DetectionResult) => {
                    const { box, label, confidence } = det;
                    
                    // Draw box with gradient stroke
                    ctx.strokeStyle = '#22c55e';
                    ctx.lineWidth = 3;
                    ctx.shadowColor = 'rgba(34, 197, 94, 0.5)';
                    ctx.shadowBlur = 8;
                    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
                    ctx.shadowBlur = 0;
                    
                    // Draw label background with rounded corners
                    const displayLabel = det.details?.vi_name ? `${det.details.vi_name} | ${label}` : label;
                    const text = `${displayLabel} (${(confidence * 100).toFixed(0)}%)`;
                    ctx.font = 'bold 16px Inter, Arial, sans-serif';
                    const textMetrics = ctx.measureText(text);
                    const textWidth = textMetrics.width;
                    const padding = 8;
                    
                    const textX = box[0];
                    const textY = box[1] > 30 ? box[1] - 30 : box[1];

                    // Rounded rectangle background
                    ctx.fillStyle = 'rgba(34, 197, 94, 0.95)';
                    ctx.beginPath();
                    ctx.roundRect(textX, textY, textWidth + padding * 2, 26, 6);
                    ctx.fill();
                    
                    // Draw text
                    ctx.fillStyle = '#000000';
                    ctx.fillText(text, textX + padding, textY + 18);
                });
            };
            img.onerror = () => {
                setError('Không thể tải ảnh. Định dạng có thể không được hỗ trợ.');
            };
            img.src = imgSrc;
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

    }, [imgSrc, detections]);

    return (
        <div className="container">
            <section id="upload" className={styles.section} aria-labelledby="upload-title">
                <h1 id="upload-title" className={styles.title}>WildLens — Nhận diện Động vật</h1>
                <p className={styles.subtitle}>Tải ảnh lên để hệ thống AI nhận diện các loài động vật trong hình ảnh của bạn.</p>

                <div 
                  className={`${styles.dropzone} dropzone ${!isMobile && isDragOver ? 'dragover' : ''}`} 
                  onDrop={!isMobile ? onDrop : undefined}
                  onDragOver={!isMobile ? onDragOver : undefined}
                  onDragLeave={!isMobile ? onDragLeave : undefined}
                  role="region"
                  aria-label={isMobile ? 'Chọn nguồn ảnh' : 'Kéo thả ảnh vào đây hoặc chọn ảnh từ máy'}
                >
                  <div className={styles.dropInner}>
                    <UploadIcon className={styles.uploadIcon} />
                    
                    {isMobile ? (
                      <>
                        <p><strong>Chọn nguồn ảnh</strong></p>
                        <div className={styles.actionsRow}>
                          <button
                            type="button"
                            className="btn primary"
                            onClick={triggerMobileCamera}
                            aria-label="Chụp bằng camera"
                          >
                            <CameraIcon />
                            Dùng camera
                          </button>
                          <label htmlFor="gallery-input" className="btn" aria-label="Chọn ảnh từ thư viện">
                            <GalleryIcon />
                            Chọn từ thư viện
                          </label>
                        </div>
                        <input
                          ref={cameraInputRef}
                          id="camera-input"
                          type="file"
                          accept="image/*"
                          capture="environment"
                          onChange={handleFileChange}
                          className="visually-hidden"
                        />
                        <input
                          ref={galleryInputRef}
                          id="gallery-input"
                          type="file"
                          accept="image/*,.jpg,.jpeg,.png,.webp,.avif,.bmp,.gif,.tif,.tiff"
                          onChange={handleFileChange}
                          className="visually-hidden"
                        />
                      </>
                    ) : (
                      <>
                        <p><strong>Kéo & thả ảnh</strong> vào đây hoặc</p>
                        <div className={styles.actionsRow}>
                          <label htmlFor="file-input" className="btn primary" aria-label="Chọn ảnh từ máy">
                            <GalleryIcon />
                            Chọn ảnh
                          </label>
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
                    <p className="muted">Hỗ trợ JPG, PNG, WebP, AVIF, BMP, GIF, TIFF</p>
                  </div>
                </div>

                {isLoading && <LoadingShimmer />}
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
                <h2 id="results-title" className={styles.sectionTitle}>Kết quả nhận diện</h2>

                <div className={styles.resultsGrid}>
                    <figure className={styles.canvasCard} aria-labelledby="figure-caption">
                        {isLoading ? (
                            <div className={styles.shimmerPlaceholder} />
                        ) : (
                            <canvas 
                              ref={canvasRef} 
                              className={styles.canvas}
                              aria-label={imgSrc ? 'Ảnh đã tải lên với khung nhận diện' : 'Chưa có ảnh để hiển thị'}
                              role="img"
                            />
                        )}
                        <figcaption id="figure-caption" className="visually-hidden">
                          Ảnh gốc và các khung bao của các đối tượng được nhận diện.
                        </figcaption>
                    </figure>

                    <div className={styles.infoBox}>
                        {!imgSrc && (
                          <div className={styles.empty}>
                            <p>Chưa có ảnh. Hãy tải ảnh ở phần trên để bắt đầu nhận diện.</p>
                          </div>
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
                                  <span className={styles.label}>({det.label})</span>
                                  <span className={styles.conf}>{(det.confidence * 100).toFixed(0)}%</span>
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

            <section id="about" className={`${styles.section} ${styles.aboutSection}`} aria-labelledby="about-title">
                <h2 id="about-title" className={styles.sectionTitle}>Về WildLens</h2>
                <p className="muted">WildLens sử dụng mô hình YOLO qua FastAPI backend để nhận diện các loài động vật trong ảnh. Trải nghiệm được tối ưu cho di động và hỗ trợ truy cập bằng bàn phím.</p>
            </section>
        </div>
    );
}