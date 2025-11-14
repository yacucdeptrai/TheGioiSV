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
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

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
        setError(e?.message || 'Cannot load history.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="container">
      <section className={styles.section} aria-labelledby="history-title">
        <h1 id="history-title" className={styles.sectionTitle}>Lịch sử kết quả</h1>
        <p className="muted">Các kết quả được lưu tạm trong {ttl} phút gần đây.</p>

        {loading && <div className={styles.banner} role="status">Đang tải lịch sử...</div>}
        {error && <div className={`${styles.banner} ${styles.error}`} role="alert">{error}</div>}

        {!loading && !error && items.length === 0 && (
          <div className={styles.empty} role="status">Chưa có kết quả gần đây.</div>
        )}

        {!loading && !error && items.length > 0 && (
          <ul style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1rem', listStyle: 'none', padding: 0, margin: 0 }}>
            {items.map(item => (
              <li key={item.id} className={styles.infoCard}>
                <Link href={`/history/${item.id}`} aria-label={`Mở chi tiết lịch sử ${item.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ display: 'grid', gap: '.5rem' }}>
                    {item.thumb_b64 ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={item.thumb_b64} alt="Ảnh thu nhỏ" style={{ width: '100%', height: 'auto', borderRadius: '8px', border: '1px solid var(--border)' }} />
                    ) : (
                      <div className={styles.empty} style={{ minHeight: 120, display: 'grid', placeItems: 'center' }}>Không có ảnh</div>
                    )}
                    <div>
                      <div style={{ fontWeight: 600 }}>{item.count} đối tượng</div>
                      <div className="muted" aria-label="Nhãn">{item.labels.slice(0, 3).join(', ')}{item.labels.length > 3 ? '…' : ''}</div>
                      <div className="muted" aria-label="Thời gian">{timeAgo(item.ts)}</div>
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
