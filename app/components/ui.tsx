// Small shared pieces: time formats, pills, switches, notices.
import type { ReactNode } from "react";

export const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

export function clock(t: number, seconds = true): string {
  const s = ((Math.floor(t) % 86400) + 86400) % 86400;
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return seconds ? `${hh}:${mm}:${ss}` : `${hh}:${mm}`;
}

export function ago(seconds: number | null): string {
  if (seconds === null) return "No reading yet";
  if (seconds < 1) return "Just now";
  if (seconds < 60) return `${Math.floor(seconds)} s ago`;
  return `${Math.floor(seconds / 60)} min ago`;
}

export function Pill({ on, onClick, children, label }: { on: boolean; onClick: () => void; children: ReactNode; label?: string }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={on} aria-label={label}
      className={`pressable inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-sm font-semibold ${on
        ? "border-accent bg-accent text-accent-ink" : "border-rule bg-surface text-ink-2 hover:bg-surface-2"}`}>
      {children}
    </button>
  );
}

export function Switch({ on, onChange, label, hint }: { on: boolean; onChange: (on: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4">
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-ink">{label}</span>
        {hint && <span className="block text-xs text-ink-3">{hint}</span>}
      </span>
      <span className="relative inline-flex shrink-0">
        <input type="checkbox" role="switch" checked={on} onChange={(e) => onChange(e.target.checked)} className="peer sr-only" />
        <span aria-hidden className={`h-7 w-12 rounded-full transition-colors ${on ? "bg-accent" : "bg-rule-strong"} peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-[var(--focus)]`} />
        <span aria-hidden className={`absolute top-1 h-5 w-5 rounded-full bg-paper shadow transition-transform ${on ? "translate-x-6" : "translate-x-1"}`} />
      </span>
    </label>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  return (
    <p className={`rounded-card px-4 py-3 text-sm ${tone === "warn" ? "bg-highlight-soft text-ink" : "bg-accent-soft text-ink"}`}>{children}</p>
  );
}

export function LineBadge() {
  return <span className="rounded-md bg-line px-1.5 py-0.5 text-[11px] font-bold text-paper">KJ</span>;
}
