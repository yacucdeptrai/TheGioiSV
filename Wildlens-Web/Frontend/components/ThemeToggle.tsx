'use client';

import { useEffect, useState } from 'react';

function getInitialTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'dark';
  const stored = localStorage.getItem('theme') as 'light' | 'dark' | null;
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

export default function ThemeToggle({
  variant = 'button',
}: {
  variant?: 'button' | 'icon';
}) {
  const [theme, setTheme] = useState<'light' | 'dark'>(getInitialTheme());
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('theme', theme);
    } catch {}
  }, [theme]);

  const toggle = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));

  const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
  const icon = theme === 'dark' ? '🌞' : '🌙';

  if (!mounted && variant === 'icon') {
    // Prevent hydration mismatch for icon-only usage
    return (
      <button
        type="button"
        aria-label="Toggle color theme"
        className="btn"
        onClick={toggle}
      >
        🌓
      </button>
    );
  }

  if (variant === 'icon') {
    return (
      <button
        type="button"
        aria-label={label}
        title={label}
        className="btn"
        onClick={toggle}
      >
        <span aria-hidden>{icon}</span>
      </button>
    );
  }

  return (
    <button type="button" className="btn" onClick={toggle} aria-label={label}>
      <span aria-hidden style={{ fontSize: '1rem' }}>{icon}</span>
      <span>{theme === 'dark' ? 'Light' : 'Dark'} mode</span>
    </button>
  );
}
