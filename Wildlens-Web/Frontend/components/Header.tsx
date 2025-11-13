'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import ThemeToggle from './ThemeToggle';

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
          <span className="brand-mark" aria-hidden>🦊</span>
          <span className="brand-text">WildLens</span>
        </Link>

        <nav className="primary-nav" aria-label="Main">
          <ul>
            <li><a href="#upload">Upload</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#about">About</a></li>
          </ul>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
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
          <li>
            <ThemeToggle />
          </li>
        </ul>
      </div>
    </header>
  );
}
