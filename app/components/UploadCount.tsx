"use client";
/**
 * Count the people in a photo the user picks. The photo is scaled down in the browser and sent only to
 * the local counter (model/aera_vision/serve.py on localhost), which replies with the count and boxes.
 * Nothing is uploaded anywhere else or stored.
 */
import { useRef, useState } from "react";
import { ImageSquare, UploadSimple } from "@phosphor-icons/react";
import { CrowdTag } from "@/components/Crowd";
import { type CountResult } from "@/components/LiveCamera";
import { COUNTER_DOWN, COUNTER_PRIVACY, COUNTER_WAKING, countImage } from "@/lib/counter";
import { VIEW_CAPACITY } from "@/lib/sim/constants";

const MAX_SIDE = 1600;

async function toJpeg(file: File): Promise<{ blob: Blob; w: number; h: number }> {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d")!.drawImage(bmp, 0, 0, w, h);
  const blob: Blob = await new Promise((r) => c.toBlob((b) => r(b!), "image/jpeg", 0.9));
  return { blob, w, h };
}

export function UploadCount() {
  const input = useRef<HTMLInputElement>(null);
  const [img, setImg] = useState<{ url: string; w: number; h: number } | null>(null);
  const [result, setResult] = useState<CountResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  async function run(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) { setError("Please choose an image file (JPG or PNG)."); return; }
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      const { blob, w, h } = await toJpeg(file);
      if (img) URL.revokeObjectURL(img.url);
      setImg({ url: URL.createObjectURL(blob), w, h });
      setResult(await countImage(blob));
    } catch {
      setError(COUNTER_DOWN);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card grid gap-3 p-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); void run(e.dataTransfer.files[0]); }}
        className={`grid gap-3 rounded-sign border border-dashed p-4 ${drag ? "border-accent bg-accent-soft" : "border-rule-strong bg-surface-2"}`}>
        {img ? (
          <figure className="relative mx-auto w-full max-w-3xl overflow-hidden rounded-sign bg-ops">
            {/* the user's own photo, shown as chosen */}
            <img src={img.url} alt="Your photo with the people the model found" width={img.w} height={img.h} className="block h-auto w-full" />
            {result && (
              <svg aria-hidden viewBox={`0 0 ${img.w} ${img.h}`} className="pointer-events-none absolute inset-0 h-full w-full">
                {result.boxes.map(([x1, y1, x2, y2], i) => (
                  <rect key={i} x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke="var(--brand-train)" strokeWidth={Math.max(2, img.w / 400)} />
                ))}
                {result.points?.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={Math.max(3, img.w / 200)} fill="var(--brand-train)" />)}
              </svg>
            )}
          </figure>
        ) : (
          <div className="flex flex-col items-center gap-2 py-8 text-center text-sm text-ink-2">
            <ImageSquare size={36} weight="light" aria-hidden className="text-ink-3" />
            Drag a photo here, or choose one from your computer.
          </div>
        )}
        <div className="flex flex-wrap items-center justify-center gap-3">
          <input ref={input} type="file" accept="image/*" className="sr-only" onChange={(e) => void run(e.target.files?.[0])} aria-label="Choose a photo" />
          <button type="button" onClick={() => input.current?.click()} disabled={busy}
            className="pressable inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2 text-sm font-semibold text-paper disabled:opacity-60">
            <UploadSimple size={17} weight="bold" aria-hidden />{img ? "Choose another photo" : "Choose a photo"}
          </button>
          {busy && <span className="text-sm text-ink-3">{COUNTER_WAKING}</span>}
        </div>
      </div>
      {error && <p className="text-sm font-semibold text-serious" role="status">{error}</p>}
      {result && (
        <p className="flex flex-wrap items-center gap-3 text-sm text-ink-2" aria-live="polite">
          <span><span className="num text-3xl font-bold text-ink">{result.count}</span> people</span>
          <CrowdTag loadPct={(result.count / VIEW_CAPACITY) * 100} />
          <span className="text-xs text-ink-3">Level as if this were half a car ({VIEW_CAPACITY} people when full) · <span className="num">{Math.round(result.ms)}</span> ms</span>
        </p>
      )}
      <p className="text-xs text-ink-3">{COUNTER_PRIVACY}</p>
    </div>
  );
}
