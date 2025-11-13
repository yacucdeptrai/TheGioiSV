'use client'; // Đánh dấu đây là Client Component

import { useState, useRef, useEffect, ChangeEvent } from 'react';
// Đảm bảo file page.module.css tồn tại trong thư mục app/
import styles from './page.module.css'; 

// --- Định nghĩa "kiểu" dữ liệu cho TypeScript ---

// 1. Định nghĩa kiểu cho 1 đối tượng được phát hiện
interface DetectionDetail {
    vi_name: string;
    habitat: string;
    lifespan: string;
    note: string;
}

interface DetectionResult {
    box: [number, number, number, number]; // [x1, y1, x2, y2]
    label: string;
    confidence: number;
    details: DetectionDetail;
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

    const canvasRef = useRef<HTMLCanvasElement>(null);

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

    // Hàm gửi ảnh đi để nhận diện
    const handleDetection = async (fileToUpload: File) => {
        if (!fileToUpload) return;

        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", fileToUpload);

        try {
            // Gọi đến backend FastAPI (đang chạy trên port 8000)
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
                setError(err.message || "Có lỗi xảy ra khi gọi API. Đảm bảo backend đang chạy.");
            } else {
                setError("Có lỗi không xác định xảy ra.");
            }
        } finally {
            setIsLoading(false);
        }
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
                ctx.drawImage(img, 0, 0);

                // 2. Vẽ các hộp Bounding Box
                detections.forEach((det: DetectionResult) => {
                    const { box, label, confidence } = det;
                    
                    // box = [x1, y1, x2, y2]
                    ctx.strokeStyle = '#00FF00'; // Màu xanh lá
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
                    
                    // Vẽ nền cho text
                    ctx.fillStyle = '#00FF00';
                    const text = `${label} (${(confidence * 100).toFixed(0)}%)`;
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
            img.src = imgSrc;
        } else {
            // Xóa canvas nếu không có ảnh
             ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

    }, [imgSrc, detections]); // Chạy lại khi 2 giá trị này thay đổi

    return (
        <main className={styles.main}>
            <h1 className={styles.title}>Nhận diện Động vật</h1>
            <p>Chụp ảnh hoặc tải ảnh lên để nhận diện</p>
            
            <input 
                type="file" 
                accept="image/*" 
                
                onChange={handleFileChange}
                className={styles.fileInput}
            />
            
            {isLoading && <p>Đang xử lý... (Backend đang chạy mô hình AI)</p>}
            {error && <p className={styles.error}>{error}</p>}

            <h2>Kết quả:</h2>
            <canvas ref={canvasRef} className={styles.canvas}></canvas>
            
            <div className={styles.infoBox}>
                {detections.length === 0 && imgSrc && !isLoading && (
                    <p>Không phát hiện thấy động vật nào trong danh sách.</p>
                )}
                
                {detections.map((det: DetectionResult, index: number) => (
                    <div key={index} className={styles.infoCard}>
                        <h3>Phát hiện: {det.details.vi_name} ({det.label})</h3>
                        <ul>
                            <li><strong>Nơi sống:</strong> {det.details.habitat}</li>
                            <li><strong>Tuổi thọ:</strong> {det.details.lifespan}</li>
                            <li><strong>Ghi chú:</strong> {det.details.note}</li>
                        </ul>
                    </div>
                ))}
            </div>
        </main>
    );
}