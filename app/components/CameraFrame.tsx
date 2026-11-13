"use client";
// One camera's view: the stored photo (faces already blurred when it was made) with what the model
// found drawn on top: boxes for a detector, dots for a density counter.
import type { BankFrame } from "@/lib/sim/engine";

export function CameraFrame({ frame, overlay = true, label, className = "" }: {
  frame: BankFrame; overlay?: boolean; label: string; className?: string;
}) {
  return (
    <figure className={`relative overflow-hidden rounded-sign bg-ops ${className}`}>
      <img src={`/demo/frames/${frame.file}`} alt={label} width={frame.w} height={frame.h}
        className="block h-full w-full object-cover" loading="lazy" />
      {overlay && (
        <svg aria-hidden viewBox={`0 0 ${frame.w} ${frame.h}`} preserveAspectRatio="xMidYMid slice"
          className="pointer-events-none absolute inset-0 h-full w-full">
          {frame.boxes?.map(([x1, y1, x2, y2], i) => (
            <rect key={i} x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke="var(--brand-train)"
              strokeWidth={Math.max(1.5, frame.w / 400)} rx={2} />
          ))}
          {frame.points?.map(([x, y, v], i) => (
            <circle key={i} cx={x} cy={y} r={Math.max(3, frame.w / 140)} fill="var(--brand-train)"
              opacity={Math.min(0.95, 0.25 + v)} />
          ))}
        </svg>
      )}
    </figure>
  );
}
