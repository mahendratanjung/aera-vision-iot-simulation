"use client";
import { useState } from "react";
import { CheckCircle, PlugsConnected, WarningCircle } from "@phosphor-icons/react";
import { Notice, Switch } from "@/components/ui";
import { deviceId } from "@/lib/sim/engine";
import { klNow, line, useSim } from "@/lib/store";

async function forwardGet(target: string, path: string, query = "") {
  const res = await fetch("/api/forward", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target, path, method: "GET", query }) });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.error ?? "AERA did not answer");
  return json as { status: number; body: unknown };
}

interface NextTrains { directions?: Record<string, { trip_id: string; arrival: number }[]> }

export default function SettingsPage() {
  const s = useSim();
  const [url, setUrl] = useState(s.send.url);
  const [token, setToken] = useState(s.send.token);
  const [busy, setBusy] = useState(false);

  async function connect(on: boolean) {
    if (!on) {
      s.setSend({ on: false, tripId: null, status: "idle", message: "" });
      s.restart();
      return;
    }
    setBusy(true);
    s.setSend({ url, token, status: "checking", message: "Checking AERA…" });
    try {
      const health = await forwardGet(url, "/api/health");
      if (health.status !== 200) throw new Error(`AERA answered ${health.status}`);
      const now = klNow();
      const nt = await forwardGet(url, `/api/stations/${s.stationId}/next-trains`, "limit=1");
      const row = (nt.body as NextTrains).directions?.[String(s.direction)]?.[0];
      if (!row) throw new Error(`AERA has no train coming to ${line.name(s.stationId)} in this direction right now`);
      s.setSend({ on: true, url, token, tripId: row.trip_id, status: "ok",
        message: `Connected. Following AERA's trip ${row.trip_id}, the next train at ${line.name(s.stationId)}.` });
      s.setSpeed(1);
      s.restart({ weekday: now.weekday, timeS: now.timeS });
    } catch (e) {
      s.setSend({ on: false, tripId: null, status: "error", message: e instanceof Error ? e.message : "AERA did not answer" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto grid max-w-2xl gap-4">
      <header>
        <p className="label">Settings</p>
        <h1 className="text-2xl font-bold tracking-tight text-ink">Send to AERA</h1>
      </header>

      <section className="card grid gap-4 p-4 sm:p-5" aria-labelledby="send-h">
        <h2 id="send-h" className="sr-only">Connection</h2>
        <Switch on={s.send.on} onChange={(on) => void connect(on)} label="Send readings to AERA"
          hint="Off: readings are only logged here. On: each reading is posted to AERA as it is made." />
        <label className="grid gap-1 text-sm font-semibold text-ink">AERA backend address
          <input value={url} onChange={(e) => setUrl(e.target.value)} disabled={s.send.on} inputMode="url" autoComplete="off"
            className="rounded-sign border border-rule bg-surface px-3 py-2 font-medium disabled:opacity-60" />
        </label>
        <label className="grid gap-1 text-sm font-semibold text-ink">Ingest token
          <input type="password" value={token} onChange={(e) => setToken(e.target.value)} disabled={s.send.on} autoComplete="off"
            className="rounded-sign border border-rule bg-surface px-3 py-2 font-medium disabled:opacity-60" />
          <span className="text-xs font-normal text-ink-3">AERA&rsquo;s AERA_INGEST_TOKEN. Kept in memory only: it is gone when you reload.</span>
        </label>
        {s.send.message && (
          <p role="status" className={`flex items-start gap-2 text-sm font-semibold ${s.send.status === "error" ? "text-critical" : "text-ink"}`}>
            {s.send.status === "error" ? <WarningCircle size={18} weight="fill" aria-hidden className="shrink-0" />
              : s.send.status === "ok" ? <CheckCircle size={18} weight="fill" aria-hidden className="shrink-0 text-good" />
              : <PlugsConnected size={18} weight="bold" aria-hidden className="shrink-0" />}
            {s.send.message}
          </p>
        )}
        {busy && <p className="text-sm text-ink-3">Connecting…</p>}
        <Notice>When sending, the simulation runs at real speed on today&rsquo;s Kuala Lumpur time and follows the next train
          AERA expects at {line.name(s.stationId)}, so AERA can match the readings to it. For a local AERA, start it with npm run dev
          in the AERA folder (address http://localhost:8000).</Notice>
      </section>

      <section className="card grid gap-2 p-4 sm:p-5" aria-labelledby="dev-h">
        <h2 id="dev-h" className="text-base font-bold text-ink">The devices</h2>
        <ul className="grid gap-1.5 text-sm text-ink-2">
          <li>One counting device per car, with two cameras: one at each end, each counting its half.</li>
          <li>Device names: <span className="num font-semibold text-ink">{[1, 2, 3, 4].map(deviceId).join(", ")}</span></li>
          <li>Counts 2 frames a second. Sends the median of the last 9 frames when it changes by 3 people or more, and at least every 30 s.</li>
          <li>Readings wait on the device without signal (up to 500) and are dropped after 2 minutes.</li>
          <li>Confidence comes from the counting error measured for the model (see Model).</li>
        </ul>
      </section>
    </div>
  );
}
