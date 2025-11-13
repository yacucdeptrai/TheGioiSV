export default function Footer() {
  return (
    <footer className="site-footer" role="contentinfo">
      <div className="container footer-inner">
        <p>
          © {new Date().getFullYear()} Sothanhtra - WildLens. Built with Next.js.
          <span className="muted"> Be kind to wildlife.</span>
        </p>
        <nav aria-label="Footer">
          <ul>
            <li><a href="#about">About</a></li>
            <li><a href="https://nextjs.org" target="_blank" rel="noreferrer noopener">Next.js</a></li>
            <li><a href="/" aria-label="Back to top">Top ↑</a></li>
          </ul>
        </nav>
      </div>
    </footer>
  );
}
