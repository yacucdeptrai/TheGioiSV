'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import styles from '../page.module.css';

interface HistoryItem {
  id: string;
  ts: string;
  labels: string[];
  count: number;
  thumb_b64?: string | null;
}

interface HistoryResponse {
  items: HistoryItem[];
  ttl_minutes: number;
}

function timeAgo(iso: string) {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const diff = Math.max(0, now - then);
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s trước`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} phút trước`;
  const h = Math.floor(m / 60);
  return `${h} giờ trước`;
}

// Clock icon
const ClockIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <polyline points="12 6 12 12 16 14"/>
  </svg>
);

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [ttl, setTtl] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const res = await fetch(`${apiUrl}/history`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`Failed to load history: ${res.status}`);
        const data: HistoryResponse = await res.json();
        setItems(data.items || []);
        setTtl(data.ttl_minutes || 30);
      } catch (e: any) {
        setError(e?.message || 'Không thể tải lịch sử.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="container">
      <section className={styles.section} aria-labelledby="history-title">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h1 id="history-title" className={styles.title}>Lịch sử kết quả</h1>
            <p className={styles.subtitle}>Các kết quả được lưu tạm trong {ttl} phút gần đây.</p>
          </div>
          <Link href="/" className="btn primary">
            ← Trang chính
          </Link>
        </div>

        {loading && (
          <div className={styles.loading}>
            <div className={styles.loadingSpinner} />
            <span>Đang tải lịch sử...</span>
          </div>
        )}
        
        {error && <div className={`${styles.banner} ${styles.error}`} role="alert">{error}</div>}

        {!loading && !error && items.length === 0 && (
          <div className={styles.empty} role="status">
            <p>Chưa có kết quả gần đây. Hãy thử nhận diện một số ảnh!</p>
          </div>
        )}

        {!loading && !error && items.length > 0 && (
          <ul className={styles.historyGrid}>
            {items.map((item) => (
              <li key={item.id} className={styles.historyCard}>
                <Link href={`/history/${item.id}`} aria-label={`Mở chi tiết lịch sử ${item.id}`}>
                  {item.thumb_b64 ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img 
                      src={item.thumb_b64} 
                      alt="Ảnh thu nhỏ" 
                      className={styles.historyCardImage}
                    />
                  ) : (
                    <div className={styles.historyCardPlaceholder}>
                      Không có ảnh
                    </div>
                  )}
                  <div className={styles.historyCardContent}>
                    <div className={styles.historyCardTitle}>
                      {item.count} đối tượng được phát hiện
                    </div>
                    <div className={styles.historyCardMeta}>
                      {item.labels.slice(0, 3).join(', ')}{item.labels.length > 3 ? '…' : ''}
                    </div>
                    <div className={styles.historyCardMeta} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <ClockIcon />
                      {timeAgo(item.ts)}
                    </div>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
