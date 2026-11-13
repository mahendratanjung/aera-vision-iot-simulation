"use client";
/**
 * Live demo with this computer's camera, shown as a normal picture (the user's choice: it is the
 * presenter's own camera, used with consent). Set blur to hide faces. Twice a second a frame goes to
 * the local counter (model/aera_vision/serve.py on localhost only), which returns boxes and a count.
 * Frames never go anywhere else and are not kept.
 */
import { useEffect, useRef, useState } from "react";
import { VideoCamera, VideoCameraSlash } from "@phosphor-icons/react";
import { CrowdTag } from "@/components/Crowd";
import { VIEW_CAPACITY } from "@/lib/sim/constants";

export const LIVE_URL = "http://localhost:8765";

export interface CountResult { count: number; boxes: number[][]; points?: number[][]; model: string; ms: number }

type Result = CountResult;

export function LiveCamera({ blur = false }: { blur?: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [on, setOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    if (!on) return;
    let stream: MediaStream | null = null;
    let timer = 0;
    let busy = false;
    let last: Result | null = null;
    let raf = 0;
    const draw = () => {
      const v = video.current, c = canvas.current;
      if (v && c && v.videoWidth) {
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        const ctx = c.getContext("2d")!;
        if (blur) ctx.filter = "blur(14px)";
        ctx.drawImage(v, 0, 0);
        ctx.filter = "none";
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--brand-train").trim();
        ctx.lineWidth = Math.max(2, c.width / 320);
        for (const [x1, y1, x2, y2] of last?.boxes ?? []) ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = ctx.strokeStyle;
        for (const [x, y] of last?.points ?? []) { ctx.beginPath(); ctx.arc(x, y, Math.max(3, c.width / 240), 0, Math.PI * 2); ctx.fill(); }
      }
      raf = requestAnimationFrame(draw);
    };
    const send = async () => {
      const v = video.current;
      if (busy || !v || !v.videoWidth) return;
      busy = true;
      try {
        const grab = document.createElement("canvas");
        grab.width = v.videoWidth;
        grab.height = v.videoHeight;
        grab.getContext("2d")!.drawImage(v, 0, 0);
        const blob: Blob = await new Promise((r) => grab.toBlob((b) => r(b!), "image/jpeg", 0.85));
        const res = await fetch(`${LIVE_URL}/count`, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
        if (!res.ok) throw new Error(String(res.status));
        last = await res.json();
        setResult(last);
        setError(null);
      } catch {
        setError("The counter is not running. Stop and run npm run dev again, then wait for \u201cCounter ready\u201d.");
      } finally {
        busy = false;
      }
    };
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((s) => {
        stream = s;
        if (video.current) { video.current.srcObject = s; void video.current.play(); }
        timer = window.setInterval(send, 500);
        raf = requestAnimationFrame(draw);
      })
      .catch(() => { setError("Camera permission was not given."); setOn(false); });
    return () => {
      window.clearInterval(timer);
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [on, blur]);

  return (
    <div className="card grid gap-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-ink-2">{on ? "Counting twice a second." : "Uses this computer\u2019s camera. Nothing is recorded."}</p>
        <button type="button" onClick={() => { setOn(!on); setResult(null); }}
          className="pressable inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2 text-sm font-semibold text-paper">
          {on ? <VideoCameraSlash size={17} weight="fill" aria-hidden /> : <VideoCamera size={17} weight="fill" aria-hidden />}
          {on ? "Stop camera" : "Start camera"}
        </button>
      </div>
      {on ? (
        <>
          <video ref={video} muted playsInline className="hidden" />
          <canvas ref={canvas} className="w-full rounded-sign bg-ops" aria-label="Live camera with the people the model found" />
        </>
      ) : (
        <div className="flex aspect-video w-full items-center justify-center rounded-sign bg-ops text-sm text-ops-ink-2">Camera off</div>
      )}
      {error && <p className="text-sm font-semibold text-serious" role="status">{error}</p>}
      {result && (
        <p className="flex flex-wrap items-center gap-2 text-sm text-ink-2" aria-live="polite">
          <span><span className="num text-2xl font-bold text-ink">{result.count}</span> people</span>
          <CrowdTag loadPct={(result.count / VIEW_CAPACITY) * 100} size="sm" />
          <span className="text-xs text-ink-3">{result.model} · <span className="num">{Math.round(result.ms)}</span> ms</span>
        </p>
      )}
    </div>
  );
}
