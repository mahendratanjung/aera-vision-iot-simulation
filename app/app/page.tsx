"use client";
/**
 * The one page: live camera, count people in a photo, example photos.
 * Model performance and hardware are explained in the report (figures in figures/).
 */
import { useEffect, useState, type ReactNode } from "react";
import { CameraFrame } from "@/components/CameraFrame";
import { CrowdTag } from "@/components/Crowd";
import { LiveCamera } from "@/components/LiveCamera";
import { UploadCount } from "@/components/UploadCount";
import { CROWD, crowdOf } from "@/lib/crowd/levels";
import { VIEW_CAPACITY } from "@/lib/sim/constants";
import type { BankFrame } from "@/lib/sim/engine";

const pctOf = (people: number) => (people / VIEW_CAPACITY) * 100;
const PER_LEVEL = [2, 2, 3, 2, 3]; // example photos per level, Quiet to Packed

export default function Home() {
  return (
    <div className="grid gap-12">
      <h1 className="sr-only">AERA Vision people counter</h1>
      <Section label="Live" title="Live camera" intro="Point your camera at a group of people to see the count update.">
        <LiveCamera />
      </Section>
      <Section label="Try it" title="Count people in a photo" intro="Choose any photo with people in it.">
        <UploadCount />
      </Section>
      <Gallery />
      <footer className="border-t border-rule pt-4 text-xs text-ink-3">Example photos: CrowdHuman (Shao et al., 2018), CC BY-NC 4.0, faces blurred.</footer>
    </div>
  );
}

function Section({ label, title, intro, children }: { label: string; title: string; intro?: ReactNode; children: ReactNode }) {
  const id = title.toLowerCase().replace(/\W+/g, "-");
  return (
    <section className="grid gap-4" aria-labelledby={id}>
      <div className="grid gap-1">
        <p className="text-xs font-semibold uppercase tracking-wider text-accent">{label}</p>
        <h2 id={id} className="text-2xl font-bold tracking-tight text-ink">{title}</h2>
        {intro && <p className="max-w-3xl text-sm text-ink-2">{intro}</p>}
      </div>
      {children}
    </section>
  );
}

/* ---- example photos: a mix of crowding levels ------------------------------------------------------ */

function pickMix(frames: BankFrame[]): BankFrame[] {
  return CROWD.flatMap((c, k) => {
    const same = frames.filter((f) => crowdOf(pctOf(f.model)).level === c.level).sort((a, z) => a.model - z.model);
    const n = Math.min(PER_LEVEL[k], same.length);
    return Array.from({ length: n }, (_, j) => same[Math.round(((j + 1) * (same.length - 1)) / (n + 1)) || 0]);
  });
}

function Gallery() {
  const [frames, setFrames] = useState<BankFrame[] | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch("/demo/bank.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((b: { frames: BankFrame[] }) => setFrames(pickMix(b.frames)))
      .catch(() => setError(true));
  }, []);
  return (
    <Section label="Examples" title="Example photos">
      {error && <p className="text-sm text-serious">The example photos are missing. In the model folder run: uv run python -m aera_vision.demo</p>}
      <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {(frames ?? []).map((f) => (
          <li key={f.id} className="overflow-hidden rounded-sign border border-rule bg-surface">
            <CameraFrame frame={f} className="aspect-3/2 rounded-none" label={`Example photo, ${f.model} people detected, faces blurred`} />
            <div className="flex items-center justify-between gap-3 border-t border-rule px-4 py-3">
              <p className="flex items-baseline gap-1.5">
                <span className="num text-2xl font-bold leading-none text-ink">{f.model}</span>
                <span className="text-sm text-ink-3">people detected</span>
              </p>
              <CrowdTag loadPct={pctOf(f.model)} size="sm" />
            </div>
          </li>
        ))}
      </ul>
    </Section>
  );
}
