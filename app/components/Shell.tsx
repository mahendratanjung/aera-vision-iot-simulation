"use client";
// Top bar with the Express logo and "Vision".
import { useId } from "react";

/** The Express mark, drawn from the theme tokens (same geometry as AERA's LogoMark and brand/build.mjs). */
export function LogoMark({ className = "h-8 w-8" }: { className?: string }) {
  const id = `aera-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const train = { fill: "var(--brand-train)" };
  return (
    <svg aria-hidden viewBox="0 0 64 64" className={`shrink-0 ${className}`} style={{ color: "var(--paper)" }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" style={{ stopColor: "var(--brand-tile)" }} />
          <stop offset="1" style={{ stopColor: "var(--brand-tile-2)" }} />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill={`url(#${id})`} />
      <rect x="8.5" y="31" width="7" height="2" rx="1" opacity=".7" style={train} />
      <rect x="5" y="35" width="9.5" height="2" rx="1" opacity=".4" style={train} />
      <path fillRule="evenodd" style={train}
        d="M17 31H41C49.25 31 54.8 33.8 56 38H17ZM40.4 32.54H45.5C48.5 32.54 50.9 33.66 52.1 34.64H40.4ZM25.4 32.6H28.8A.8.8 0 0 1 29.6 33.4V34A.8.8 0 0 1 28.8 34.8H25.4A.8.8 0 0 1 24.6 34V33.4A.8.8 0 0 1 25.4 32.6Z" />
      <path d="M12 48L24 15H31L43 48H36L27.5 24.63L19 48Z" fill="currentColor" />
    </svg>
  );
}

export function Logo() {
  return (
    <span className="flex items-center gap-2">
      <LogoMark className="h-8 w-8 drop-shadow-sm" />
      <span className="text-lg font-extrabold tracking-tight text-ink">AERA <span className="text-accent">Vision</span></span>
    </span>
  );
}

export function TopBar() {
  return (
    <header className="sticky top-0 z-30 border-b border-rule bg-paper/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-4 px-4">
        <Logo />
      </div>
    </header>
  );
}
