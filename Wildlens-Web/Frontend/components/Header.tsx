'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import ThemeToggle from './ThemeToggle';

// Logo icon
const LogoIcon = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path 
      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" 
      fill="url(#logo-gradient)"
      fillOpacity="0.15"
    />
    <path 
      d="M12 6c-2.21 0-4 1.79-4 4 0 1.48.81 2.77 2 3.46V16c0 .55.45 1 1 1h2c.55 0 1-.45 1-1v-2.54c1.19-.69 2-1.98 2-3.46 0-2.21-1.79-4-4-4z" 
      fill="url(#logo-gradient)"
    />
    <circle cx="10" cy="9.5" r="0.75" fill="white"/>
    <circle cx="14" cy="9.5" r="0.75" fill="white"/>
    <defs>
      <linearGradient id="logo-gradient" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
        <stop stopColor="#0ea5e9"/>
        <stop offset="1" stopColor="#22c55e"/>
      </linearGradient>
    </defs>
  </svg>
);

export default function Header() {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    function onClick(e: MouseEvent) {
      if (open && menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('click', onClick);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('click', onClick);
    };
  }, [open]);

  return (
    <header className="site-header">
      <div className="container header-inner">
        <Link href="/" className="brand" aria-label="WildLens home">
          <LogoIcon />
          <span className="brand-text">WildLens</span>
        </Link>

        <nav className="primary-nav" aria-label="Main">
          <ul>
            <li><a href="#upload">Upload</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#about">About</a></li>
            <li><Link href="/history">History</Link></li>
          </ul>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
          <ThemeToggle variant="icon" />
          <button
            className="hamburger"
            aria-label="Open menu"
            aria-controls="mobile-menu"
            aria-expanded={open}
            onClick={() => setOpen(v => !v)}
          >
            <span className="hamburger-box" aria-hidden>
              <span className="hamburger-inner" />
            </span>
          </button>
        </div>
      </div>

      <div
        id="mobile-menu"
        className={`mobile-drawer ${open ? 'open' : ''}`}
        ref={menuRef}
        role="dialog"
        aria-modal="true"
        aria-label="Mobile Menu"
      >
        <ul>
          <li><a href="#upload" onClick={() => setOpen(false)}>Upload</a></li>
          <li><a href="#results" onClick={() => setOpen(false)}>Results</a></li>
          <li><a href="#about" onClick={() => setOpen(false)}>About</a></li>
          <li><Link href="/history" onClick={() => setOpen(false)}>History</Link></li>
          <li>
            <ThemeToggle />
          </li>
        </ul>
      </div>
    </header>
  );
}
