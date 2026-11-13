"use client";
import { CheckCircle, LockKey, WarningCircle } from "@phosphor-icons/react";
import { CameraFrame } from "@/components/CameraFrame";
import { CrowdTag, TrainCars } from "@/components/Crowd";
import { CountChart } from "@/components/CountChart";
import { LiveCamera } from "@/components/LiveCamera";
import { clock, Notice, Pill } from "@/components/ui";
import { checkCarCount } from "@/lib/device/contract";
import { CAR_CAPACITY, VIEW_CAPACITY } from "@/lib/sim/constants";
import { useSim } from "@/lib/store";
import type { CarState, LogEntry } from "@/lib/sim/engine";

const KIND: Record<LogEntry["kind"], string> = {
  sent: "Sent", rejected: "Rejected by AERA", dropped: "Dropped", offline: "Waiting: no signal", retry: "Waiting: AERA busy",
};

export default function CameraPage() {
  useSim((s) => s.tick);
  const { engine, bank, bankError, selectedCar, selectCar, logs } = useSim();
  if (!engine || !bank) return <div className="mx-auto max-w-xl pt-10"><Notice tone="warn">{bankError ?? "Starting the cameras…"}</Notice></div>;
  const car = engine.cars[selectedCar - 1];
  const carLogs = logs.filter((l) => l.car === car.car).slice(0, 12);
  const isDensity = bank.model.kind === "density";
  return (
    <div className="grid gap-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label">Cameras · Faces blurred</p>
          <h1 className="text-2xl font-bold tracking-tight text-ink">{car.car === 1 ? "Front car" : `Car ${car.car}`}</h1>
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Car">
          {engine.cars.map((c) => <Pill key={c.car} on={c.car === selectedCar} onClick={() => selectCar(c.car)}>{c.car === 1 ? "Front" : `Car ${c.car}`}</Pill>)}
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px] lg:items-start">
        <div className="grid min-w-0 gap-4">
          <section className="card grid gap-3 p-3.5 sm:p-4" aria-label="Camera views">
            <div className="grid gap-3 sm:grid-cols-2">
              {car.views.map((v, i) => (
                <div key={i} className="grid gap-1.5">
                  <CameraFrame frame={v.frame} className="aspect-[4/3]"
                    label={`${i === 0 ? "Front" : "Back"} half of car ${car.car}, faces blurred, with what the model found`} />
                  <p className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <span className="font-semibold text-ink">{i === 0 ? "Front half" : "Back half"}</span>
                    <span className="text-ink-2">Model <span className="num font-bold text-ink">{v.frame.model}</span> · Labelled <span className="num font-semibold">{v.frame.truth}</span></span>
                  </p>
                  <CrowdTag loadPct={(v.frame.model / VIEW_CAPACITY) * 100} size="sm" />
                </div>
              ))}
            </div>
            <p className="text-xs text-ink-3">
              {isDensity ? "Dots show where the model expects people (density)." : "Boxes show the people the model found."} Model: {bank.model.label}.
              {" "}Photos: {bank.attribution}
            </p>
          </section>

          <Leaves car={car} className="grid lg:hidden" />

          <section className="card grid gap-3 p-4" aria-labelledby="chart-h">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 id="chart-h" className="text-base font-bold text-ink">Count over time</h2>
              <CrowdTag loadPct={((car.lastValue ?? car.model) / CAR_CAPACITY) * 100} showPct />
            </div>
            <CountChart points={engine.history[car.car - 1]} />
          </section>

          <LiveCamera />
        </div>

        <div className="grid gap-4">
          <Leaves car={car} className="hidden lg:grid" />

          <section className="card grid gap-2 p-4" aria-labelledby="log-h">
            <h2 id="log-h" className="text-base font-bold text-ink">Send log</h2>
            {carLogs.length === 0 ? <p className="text-sm text-ink-3">Nothing yet. Press Play on the Train page.</p> : (
              <ol className="grid divide-y divide-rule text-sm">
                {carLogs.map((l, i) => (
                  <li key={i} className="flex items-center justify-between gap-2 py-1.5">
                    <span className="num text-ink-3">{clock(l.at - (engine.opts.dateBase ?? 0))}</span>
                    <span className="min-w-0 flex-1 truncate font-semibold text-ink">
                      {l.kind === "sent" && l.mode === "log" ? "Logged" : KIND[l.kind]}{l.status ? ` (${l.status})` : ""}{l.detail ? ` · ${l.detail}` : ""}
                    </span>
                    <span className="num text-ink-2">#{l.seq}{l.msg?.count !== undefined ? ` · ${l.msg.count}` : ""}</span>
                  </li>
                ))}
              </ol>
            )}
          </section>

          <section className="card p-4">
            <TrainCars cars={engine.cars.map((c) => ((c.lastValue ?? c.model) / CAR_CAPACITY) * 100)} selected={selectedCar} onSelect={selectCar} />
          </section>
        </div>
      </div>
    </div>
  );
}

function Leaves({ car, className }: { car: CarState; className: string }) {
  const last = car.lastReading;
  const problems = last ? checkCarCount(last as unknown as Record<string, unknown>) : [];
  const json = last ? JSON.stringify(last, null, 2) : "";
  return (
    <section className={`gap-3 rounded-card bg-ops p-4 text-ops-ink shadow-card ${className}`} aria-label="What leaves the car">
      <h2 className="flex items-center gap-2 text-base font-bold"><LockKey size={18} weight="bold" aria-hidden />What leaves the car</h2>
      <p className="text-sm text-ops-ink-2">The device counts on board and sends only this. No photo, no face, no position of anyone.</p>
      {last ? (
        <>
          <pre className="num overflow-x-auto rounded-sign bg-ops-2 p-3 text-[13px] leading-relaxed" aria-label="Last reading as JSON">{json}</pre>
          <p className="flex items-start gap-1.5 text-sm font-semibold">
            {problems.length === 0
              ? <><CheckCircle size={17} weight="fill" aria-hidden className="shrink-0 text-crowd-empty" /><span>Matches AERA&rsquo;s CarCount contract<span className="block text-xs font-normal text-ops-ink-2"><span className="num">{new TextEncoder().encode(JSON.stringify(last)).length}</span> bytes per reading</span></span></>
              : <><WarningCircle size={17} weight="fill" aria-hidden />{problems.join("; ")}</>}
          </p>
        </>
      ) : <p className="text-sm text-ops-ink-2">The first reading comes after 9 camera frames (4.5 s).</p>}
      <dl className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-sign bg-ops-2 p-2"><dt className="text-ops-ink-2">Sent</dt><dd className="num text-lg font-bold">{car.sent}</dd></div>
        <div className="rounded-sign bg-ops-2 p-2"><dt className="text-ops-ink-2">Waiting</dt><dd className="num text-lg font-bold">{car.queue}</dd></div>
        <div className="rounded-sign bg-ops-2 p-2"><dt className="text-ops-ink-2">Dropped</dt><dd className="num text-lg font-bold">{car.dropped}</dd></div>
      </dl>
      <p className="text-xs text-ops-ink-2">Photos sent: <span className="num font-semibold text-ops-ink">0</span> · Device {car.deviceId}</p>
    </section>
  );
}
