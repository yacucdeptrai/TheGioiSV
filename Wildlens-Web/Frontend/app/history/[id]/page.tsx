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
        // draw boxes
        (record.detections || []).forEach((det) => {
          const { box, label, confidence } = det;
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 3;
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);

          ctx.fillStyle = '#00FF00';
          const displayLabel = det.details?.vi_name ? `${det.details.vi_name} | ${label}` : label;
          const text = `${displayLabel} (${(confidence * 100).toFixed(0)}%)`;
          ctx.font = '18px Arial';
          const textWidth = ctx.measureText(text).width;
          const textX = box[0];
          const textY = box[1] > 20 ? box[1] - 20 : box[1];
          ctx.fillRect(textX, textY, textWidth + 4, 20);
          ctx.fillStyle = '#000000';
          ctx.fillText(text, textX + 2, textY + 15);
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
        <h1 id="history-detail-title" className={styles.sectionTitle}>Chi tiết lịch sử</h1>

        <div className={styles.actionsRow}>
          <Link className="btn" href="/history">← Quay lại Lịch sử</Link>
          <Link className="btn" href="/">Trang chính</Link>
        </div>

        {loading && <div className={styles.banner} role="status">Đang tải...</div>}
        {error && <div className={`${styles.banner} ${styles.error}`} role="alert">{error}</div>}

        {record && (
          <div className={styles.resultsGrid}>
            <figure className={styles.canvasCard}>
              <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label="Ảnh và khung nhận diện" />
            </figure>
            <div className={styles.infoBox}>
              <div className="muted">Thời gian: {new Date(record.ts).toLocaleString()}</div>
              {record.detections.length === 0 && (
                <div className={styles.empty}><p>Không phát hiện thấy động vật nào.</p></div>
              )}
              {record.detections.map((det, idx) => (
                <article key={idx} className={styles.infoCard}>
                  <h3>
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
        )}
      </section>
    </div>
  );
}
