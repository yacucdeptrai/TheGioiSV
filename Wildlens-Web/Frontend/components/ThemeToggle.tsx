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

// Sun icon
const SunIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5"/>
    <line x1="12" y1="1" x2="12" y2="3"/>
    <line x1="12" y1="21" x2="12" y2="23"/>
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
    <line x1="1" y1="12" x2="3" y2="12"/>
    <line x1="21" y1="12" x2="23" y2="12"/>
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
  </svg>
);

// Moon icon
const MoonIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);

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

  if (!mounted && variant === 'icon') {
    return (
      <button
        type="button"
        aria-label="Toggle color theme"
        className="btn"
        onClick={toggle}
        style={{ padding: '0.6rem' }}
      >
        <MoonIcon />
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
        style={{ 
          padding: '0.6rem',
          transition: 'transform 0.3s ease'
        }}
      >
        <span 
          aria-hidden 
          style={{ 
            display: 'flex',
            transition: 'transform 0.3s ease',
            transform: theme === 'dark' ? 'rotate(0deg)' : 'rotate(180deg)'
          }}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </span>
      </button>
    );
  }

  return (
    <button 
      type="button" 
      className="btn" 
      onClick={toggle} 
      aria-label={label}
      style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
    >
      <span aria-hidden style={{ display: 'flex' }}>
        {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
      </span>
      <span>{theme === 'dark' ? 'Light' : 'Dark'} mode</span>
    </button>
  );
}
