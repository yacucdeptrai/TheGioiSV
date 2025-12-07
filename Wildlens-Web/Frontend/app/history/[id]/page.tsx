'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import styles from '../../page.module.css';

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

interface HistoryRecord {
  id: string;
  ts: string;
  detections: DetectionResult[];
  image_b64?: string | null;
}

// Arrow icon
const ArrowLeftIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12"/>
    <polyline points="12 19 5 12 12 5"/>
  </svg>
);

// Home icon
const HomeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
    <polyline points="9 22 9 12 15 12 15 22"/>
  </svg>
);

export default function HistoryDetailPage() {
  const params = useParams<{ id: string }>();
  const recordId = params?.id as string;

  const [record, setRecord] = useState<HistoryRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!recordId) return;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const res = await fetch(`${apiUrl}/history/${recordId}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`Không tìm thấy bản ghi hoặc đã hết hạn (${res.status}).`);
        const data: HistoryRecord = await res.json();
        setRecord(data);
      } catch (e: any) {
        setError(e?.message || 'Không thể tải bản ghi.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [recordId]);

  // Draw image and boxes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (record?.image_b64) {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        try {
          ctx.drawImage(img, 0, 0);
        } catch (e) {
          // ignore
        }
        // Draw boxes with enhanced styling
        (record.detections || []).forEach((det) => {
          const { box, label, confidence } = det;
          
          ctx.strokeStyle = '#22c55e';
          ctx.lineWidth = 3;
          ctx.shadowColor = 'rgba(34, 197, 94, 0.5)';
          ctx.shadowBlur = 8;
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          ctx.shadowBlur = 0;

          const displayLabel = det.details?.vi_name ? `${det.details.vi_name} | ${label}` : label;
          const text = `${displayLabel} (${(confidence * 100).toFixed(0)}%)`;
          ctx.font = 'bold 16px Inter, Arial, sans-serif';
          const textWidth = ctx.measureText(text).width;
          const padding = 8;
          const textX = box[0];
          const textY = box[1] > 30 ? box[1] - 30 : box[1];
          
          ctx.fillStyle = 'rgba(34, 197, 94, 0.95)';
          ctx.beginPath();
          ctx.roundRect(textX, textY, textWidth + padding * 2, 26, 6);
          ctx.fill();
          
          ctx.fillStyle = '#000000';
          ctx.fillText(text, textX + padding, textY + 18);
        });
      };
      img.src = record.image_b64;
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [record]);

  return (
    <div className="container">
      <section className={styles.section} aria-labelledby="history-detail-title">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
          <h1 id="history-detail-title" className={styles.sectionTitle}>Chi tiết lịch sử</h1>
          <div className={styles.actionsRow}>
            <Link className="btn" href="/history">
              <ArrowLeftIcon />
              Quay lại
            </Link>
            <Link className="btn primary" href="/">
              <HomeIcon />
              Trang chính
            </Link>
          </div>
        </div>

        {loading && (
          <div className={styles.loading}>
            <div className={styles.loadingSpinner} />
            <span>Đang tải...</span>
          </div>
        )}
        {error && <div className={`${styles.banner} ${styles.error}`} role="alert">{error}</div>}

        {record && (
          <div className={styles.resultsGrid}>
            <figure className={styles.canvasCard}>
              <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label="Ảnh và khung nhận diện" />
            </figure>
            <div className={styles.infoBox}>
              <div className="muted" style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
                Thời gian: {new Date(record.ts).toLocaleString('vi-VN')}
              </div>
              {record.detections.length === 0 && (
                <div className={styles.empty}><p>Không phát hiện thấy động vật nào.</p></div>
              )}
              {record.detections.map((det, idx) => (
                <article key={idx} className={styles.infoCard}>
                  <h3>
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
        )}
      </section>
    </div>
  );
}
