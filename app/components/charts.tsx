"use client";
/**
 * Small SVG charts in the house style (dataviz method): one axis, 2px lines, recessive hairline grid,
 * a legend for two or more series plus a direct end label, a hover tooltip, and a table view.
 * Series colours in fixed order: --accent, --highlight, --accent-2 (validated together for colour-blind
 * and normal-vision separation). A reference line (e.g. the labelled truth) is dashed neutral ink.
 */
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export const SERIES = ["var(--accent)", "var(--highlight)", "var(--accent-2)"];
export const REFERENCE = "var(--ink-3)";

export interface Series { name: string; color: string; dashed?: boolean; points: [number, number][] }

const fmt = (v: number, d = 0) => v.toLocaleString("en-MY", { maximumFractionDigits: d, minimumFractionDigits: d });

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

export function Legend({ items }: { items: { name: string; color: string; dashed?: boolean }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map((s) => (
        <li key={s.name} className="inline-flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden><line x1="1" y1="4" x2="17" y2="4" stroke={s.color} strokeWidth="2.5" strokeLinecap="round" strokeDasharray={s.dashed ? "4 3" : undefined} /></svg>
          {s.name}
        </li>
      ))}
    </ul>
  );
}

export function TableView({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    <details className="text-xs">
      <summary className="cursor-pointer font-semibold text-accent">Show as table</summary>
      <div className="mt-2 max-h-64 overflow-auto">
        <table className="num w-full text-left">
          <caption className="sr-only">{caption}</caption>
          <thead><tr>{head.map((h) => <th key={h} className="border-b border-rule py-1 pr-3 font-semibold text-ink-2">{h}</th>)}</tr></thead>
          <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="border-b border-rule py-1 pr-3 text-ink">{c}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </details>
  );
}

/** Width of an element in CSS pixels, so SVG text stays at its real size on phones. */
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Line chart with a crosshair tooltip. x and y are numbers; formatters turn them into text. */
export function LineChart({ series, xLabel, yLabel, xFormat = (v) => fmt(v), yFormat = (v) => fmt(v), height = 220,
  xDomain, yDomain, summary, extra, marker, endLabels = true }: {
  series: Series[]; xLabel: string; yLabel: string; xFormat?: (v: number) => string; yFormat?: (v: number) => string;
  height?: number; xDomain?: [number, number]; yDomain?: [number, number]; summary: string; extra?: ReactNode;
  marker?: { x: number; y: number; label: string }; endLabels?: boolean;
}) {
  const [box, W] = useWidth(640);
  const H = height, L = 44, R = 52, T = 12, B = 34;
  const all = series.flatMap((s) => s.points);
  const [x0, x1] = xDomain ?? [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
  const [y0, y1] = yDomain ?? [0, niceMax(Math.max(1, ...all.map((p) => p[1])))];
  const sx = (x: number) => L + ((x - x0) / Math.max(1e-9, x1 - x0)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - y0) / Math.max(1e-9, y1 - y0)) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => y0 + f * (y1 - y0));
  const xt = [0, 0.5, 1].map((f) => x0 + f * (x1 - x0));
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const id = useId();
  const nearest = (s: Series, x: number) => s.points.reduce((b, p) => (Math.abs(p[0] - x) < Math.abs(b[0] - x) ? p : b), s.points[0]);
  const onMove = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const x = x0 + (((e.clientX - r.left) / r.width) * W - L) / (W - L - R) * (x1 - x0);
    setHover(Math.min(x1, Math.max(x0, x)));
  };
  return (
    <figure className="grid gap-2" aria-labelledby={`${id}-s`}>
      {series.length > 1 && <Legend items={series} />}
      <div className="relative" ref={box}>
        <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="block w-full touch-none" role="img" aria-label={summary}
          onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--rule)" strokeWidth="1" />
              <text x={L - 6} y={sy(t) + 4} textAnchor="end" className="num" fontSize="11" fill="var(--ink-3)">{yFormat(t)}</text>
            </g>
          ))}
          {xt.map((t, i) => (
            <text key={i} x={sx(t)} y={H - 14} textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"} className="num" fontSize="11" fill="var(--ink-3)">{xFormat(t)}</text>
          ))}
          <text x={(L + W - R) / 2} y={H - 1} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{xLabel}</text>
          <text x={10} y={T + 2} fontSize="11" fill="var(--ink-3)" transform={`rotate(-90 10 ${T + 2})`} textAnchor="end">{yLabel}</text>
          {series.map((s) => s.points.length > 0 && (
            <g key={s.name}>
              <polyline fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                strokeDasharray={s.dashed ? "5 4" : undefined} points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
              {endLabels && (() => { const p = s.points[s.points.length - 1]; return (
                <>
                  <circle cx={sx(p[0])} cy={sy(p[1])} r="4" fill={s.color} stroke="var(--surface)" strokeWidth="2" />
                  <text x={sx(p[0]) + 8} y={sy(p[1]) + 4} fontSize="11" fontWeight="600" fill="var(--ink-2)" className="num">{yFormat(p[1])}</text>
                </>
              ); })()}
            </g>
          ))}
          {marker && (
            <g>
              <circle cx={sx(marker.x)} cy={sy(marker.y)} r="5" fill="var(--highlight)" stroke="var(--surface)" strokeWidth="2" />
              <text x={sx(marker.x) + (sx(marker.x) > W * 0.6 ? -8 : 8)} y={sy(marker.y) - 8} textAnchor={sx(marker.x) > W * 0.6 ? "end" : "start"}
                fontSize="11" fontWeight="700" fill="var(--ink)">{marker.label}</text>
            </g>
          )}
          {hover !== null && (
            <g>
              <line x1={sx(hover)} x2={sx(hover)} y1={T} y2={H - B} stroke="var(--rule-strong)" strokeWidth="1" />
              {series.map((s) => { const p = nearest(s, hover); return p && <circle key={s.name} cx={sx(p[0])} cy={sy(p[1])} r="4" fill={s.color} stroke="var(--surface)" strokeWidth="2" />; })}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="pointer-events-none absolute top-1 rounded-sign border border-rule bg-surface px-2.5 py-1.5 text-xs shadow-card"
            style={{ left: `${Math.min(70, (sx(hover) / W) * 100)}%` }}>
            <p className="num font-semibold text-ink">{xFormat(nearest(series[0], hover)?.[0] ?? hover)}</p>
            {series.map((s) => { const p = nearest(s, hover); return p && (
              <p key={s.name} className="flex items-center gap-1.5 text-ink-2">
                <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />{s.name}: <span className="num font-semibold text-ink">{yFormat(p[1])}</span>
              </p>
            ); })}
          </div>
        )}
      </div>
      <figcaption id={`${id}-s`} className="text-xs text-ink-3">{summary}</figcaption>
      {extra}
    </figure>
  );
}

/** Predicted against true, one dot per image, with the "perfect" diagonal. */
export function Scatter({ points, xLabel, yLabel, summary, max }: {
  points: [number, number][]; xLabel: string; yLabel: string; summary: string; max: number;
}) {
  const W = 360, H = 320, L = 40, R = 12, T = 12, B = 36;
  const m = niceMax(max);
  const sx = (x: number) => L + (x / m) * (W - L - R);
  const sy = (y: number) => T + (1 - y / m) * (H - T - B);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const ticks = [0, 0.5, 1].map((f) => f * m);
  return (
    <figure className="grid gap-2">
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="block w-full max-w-[420px]" role="img" aria-label={summary} onPointerLeave={() => setHover(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--rule)" />
              <text x={L - 6} y={sy(t) + 4} textAnchor="end" fontSize="11" fill="var(--ink-3)" className="num">{fmt(t)}</text>
              <text x={sx(t)} y={H - 20} textAnchor="middle" fontSize="11" fill="var(--ink-3)" className="num">{fmt(t)}</text>
            </g>
          ))}
          <line x1={sx(0)} y1={sy(0)} x2={sx(m)} y2={sy(m)} stroke={REFERENCE} strokeDasharray="5 4" strokeWidth="1.5" />
          <text x={sx(m) - 4} y={sy(m) + 14} textAnchor="end" fontSize="11" fill="var(--ink-3)">Perfect count</text>
          {points.map(([x, y], i) => (
            <circle key={i} cx={sx(Math.min(x, m))} cy={sy(Math.min(y, m))} r="3" fill="var(--accent)" fillOpacity="0.35"
              onPointerEnter={() => setHover([x, y])} />
          ))}
          {hover && <circle cx={sx(hover[0])} cy={sy(hover[1])} r="5" fill="var(--accent)" stroke="var(--surface)" strokeWidth="2" />}
          <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{xLabel}</text>
          <text x={10} y={T + 2} fontSize="11" fill="var(--ink-3)" transform={`rotate(-90 10 ${T + 2})`} textAnchor="end">{yLabel}</text>
        </svg>
        {hover && (
          <div className="pointer-events-none absolute left-12 top-2 rounded-sign border border-rule bg-surface px-2.5 py-1.5 text-xs shadow-card">
            Labelled <span className="num font-semibold">{fmt(hover[0])}</span> · Counted <span className="num font-semibold">{fmt(hover[1], 1)}</span>
          </div>
        )}
      </div>
      <figcaption className="text-xs text-ink-3">{summary}</figcaption>
    </figure>
  );
}

/** Horizontal bars by group (e.g. by crowding level), up to three series, values at the bar tips. */
export function GroupedBars({ groups, series, format, summary, zeroCentred = false }: {
  groups: { label: ReactNode; key: string }[]; series: { name: string; color: string; values: (number | null)[] }[];
  format: (v: number) => string; summary: string; zeroCentred?: boolean;
}) {
  const vals = series.flatMap((s) => s.values.filter((v): v is number => v !== null && Number.isFinite(v)));
  const max = niceMax(Math.max(1, ...vals.map(Math.abs)));
  const bar = 12;
  return (
    <figure className="grid gap-2">
      {series.length > 1 && <Legend items={series} />}
      <div role="img" aria-label={summary} className="grid gap-3">
        {groups.map((g, gi) => (
          <div key={g.key} className="grid grid-cols-[112px_minmax(0,1fr)] items-center gap-2">
            <div className="text-xs">{g.label}</div>
            <div className="grid gap-[2px]">
              {series.map((s) => {
                const v = s.values[gi];
                const ok = v !== null && Number.isFinite(v);
                const w = ok ? (Math.abs(v!) / max) * (zeroCentred ? 50 : 100) : 0;
                const left = zeroCentred ? (v! < 0 ? 50 - w : 50) : 0;
                return (
                  <div key={s.name} className="relative flex h-[14px] items-center" title={`${s.name}: ${ok ? format(v!) : "no images"}`}>
                    {zeroCentred && <span aria-hidden className="absolute left-1/2 top-0 h-full w-px bg-rule-strong" />}
                    {ok && <span aria-hidden className="absolute h-[12px]" style={{ left: `${left}%`, width: `${Math.max(w, 0.5)}%`, height: bar, background: s.color,
                      borderRadius: zeroCentred ? (v! < 0 ? "4px 0 0 4px" : "0 4px 4px 0") : "0 4px 4px 0" }} />}
                    <span className="num absolute text-[11px] font-semibold text-ink-2"
                      style={zeroCentred ? (v! < 0 ? { right: `${100 - left + 1}%` } : { left: `${left + w + 1}%` }) : { left: `calc(${w}% + 4px)` }}>
                      {ok ? format(v!) : "n/a"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <figcaption className="text-xs text-ink-3">{summary}</figcaption>
    </figure>
  );
}
