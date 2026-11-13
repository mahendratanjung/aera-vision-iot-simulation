"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowClockwise, CellSignalFull, CellSignalSlash, LockKey, Pause, Play, VideoCamera } from "@phosphor-icons/react";
import { CrowdMeter, CrowdTag, TrainCars } from "@/components/Crowd";
import { CameraFrame } from "@/components/CameraFrame";
import { ago, clock, DAYS, LineBadge, Notice, Pill, Switch } from "@/components/ui";
import { crowdOf } from "@/lib/crowd/levels";
import { SCENARIOS, type CarState } from "@/lib/sim/engine";
import { CAR_CAPACITY } from "@/lib/sim/constants";
import { line, useSim, type Speed } from "@/lib/store";

const pctOf = (people: number) => (people / CAR_CAPACITY) * 100;
/** What AERA has been told about the car: the last reading sent, or the live count before the first one. */
const reported = (c: CarState) => c.lastValue ?? c.model;

export default function TrainPage() {
  useSim((s) => s.tick);
  const { engine, bankError } = useSim();
  if (!engine) {
    return <div className="mx-auto max-w-xl pt-10"><Notice tone={bankError ? "warn" : "info"}>{bankError ?? "Starting the cameras…"}</Notice></div>;
  }
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px] lg:items-start">
      <div className="grid min-w-0 gap-4">
        <TrainHeader />
        <section aria-labelledby="cars-h" className="grid gap-3">
          <h2 id="cars-h" className="sr-only">Cars</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {engine.cars.map((c) => <CarCard key={c.car} car={c} now={engine.t} />)}
          </div>
        </section>
        <Notice><LockKey size={16} weight="bold" aria-hidden className="mr-1.5 inline align-[-3px]" />
          Only numbers leave the train. Photos stay on the device and are never stored or sent. <Link className="font-semibold text-accent underline-offset-2 hover:underline" href="/camera">See what is sent</Link></Notice>
      </div>
      <Controls />
    </div>
  );
}

function TrainHeader() {
  const { engine, running, speed, setRunning, setSpeed, selectedCar, selectCar, restart } = useSim();
  const router = useRouter();
  if (!engine) return null;
  const tr = engine.train;
  const dir = line.dir(engine.direction);
  const where = tr.phase === "ended" ? `Arrived at ${engine.stationName(tr.stopIndex)}`
    : tr.phase === "dwelling" ? `At ${engine.stationName(tr.stopIndex)}`
    : `${engine.stationName(tr.stopIndex)} to ${engine.stationName(tr.nextIndex!)}`;
  const cars = engine.cars.map((c) => pctOf(reported(c)));
  const total = engine.cars.reduce((a, c) => a + reported(c), 0);
  return (
    <section className="card-raised grid gap-4 p-4 sm:p-5" aria-labelledby="train-h">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="label flex items-center gap-1.5"><LineBadge /> Kelana Jaya Line · Train 07</p>
          <h1 id="train-h" className="mt-1 text-xl font-bold tracking-tight text-ink sm:text-2xl">To {dir.headsign}</h1>
          <p className="text-sm text-ink-2" aria-live="polite">{where}</p>
        </div>
        <div className="text-right">
          <p className="num text-2xl font-bold text-ink">{clock(engine.t)}</p>
          <p className="text-xs text-ink-3">{DAYS[engine.weekday]} · Simulated time</p>
        </div>
      </div>
      <TrainCars cars={cars} selected={selectedCar} onSelect={(c) => { selectCar(c); router.push("/camera"); }} />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-ink-2"><span className="num font-bold text-ink">{total}</span> people counted on board ·{" "}
          <CrowdTag loadPct={(total / (CAR_CAPACITY * 4)) * 100} size="sm" /></p>
        <div className="flex flex-wrap items-center gap-2">
          {tr.phase === "ended" ? (
            <Pill on onClick={() => { restart(); setRunning(true); }}><ArrowClockwise size={16} weight="bold" aria-hidden />Run again</Pill>
          ) : (
            <Pill on={running} onClick={() => setRunning(!running)} label={running ? "Pause" : "Play"}>
              {running ? <Pause size={16} weight="fill" aria-hidden /> : <Play size={16} weight="fill" aria-hidden />}{running ? "Pause" : "Play"}
            </Pill>
          )}
          <span className="inline-flex gap-1" role="group" aria-label="Speed">
            {([1, 5, 20] as Speed[]).map((s) => <Pill key={s} on={speed === s} onClick={() => setSpeed(s)} label={`Speed ${s} times`}><span className="num">{s}×</span></Pill>)}
          </span>
        </div>
      </div>
    </section>
  );
}

function CarCard({ car, now }: { car: CarState; now: number }) {
  const { offline, selectCar } = useSim();
  const router = useRouter();
  const value = reported(car);
  const lv = crowdOf(pctOf(value));
  return (
    <article className="card grid gap-3 p-3.5" aria-label={`Car ${car.car}`}>
      <button type="button" onClick={() => { selectCar(car.car); router.push("/camera"); }}
        className="pressable grid grid-cols-2 gap-1.5 text-left" aria-label={`Open the cameras of car ${car.car}`}>
        {car.views.map((v, i) => (
          <CameraFrame key={i} frame={v.frame} overlay={false} className="aspect-[4/3]"
            label={`Car ${car.car}, ${i === 0 ? "front" : "back"} camera, faces blurred`} />
        ))}
      </button>
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="label">{car.car === 1 ? "Front car" : `Car ${car.car}`}</p>
          <p className="text-ink"><span className="num text-3xl font-bold">{value}</span> <span className="text-sm text-ink-2">people</span></p>
        </div>
        <div className="grid justify-items-end gap-1.5">
          <CrowdTag loadPct={pctOf(value)} />
          <CrowdMeter level={lv.level} size="sm" />
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <dt className="text-ink-3">Labelled in photos</dt><dd className="num text-right font-semibold text-ink">{car.truth}</dd>
        <dt className="text-ink-3">Confidence</dt><dd className="num text-right font-semibold text-ink">{car.confidence.toFixed(2)}</dd>
        <dt className="text-ink-3">Last reading</dt><dd className="text-right font-semibold text-ink">{ago(car.lastReadingAt === null ? null : now - car.lastReadingAt)}</dd>
        <dt className="text-ink-3">Waiting to send</dt><dd className="num text-right font-semibold text-ink">{car.queue}</dd>
      </dl>
      <p className={`inline-flex items-center gap-1.5 text-xs font-semibold ${offline ? "text-serious" : "text-good"}`}>
        {offline ? <CellSignalSlash size={15} weight="bold" aria-hidden /> : <CellSignalFull size={15} weight="bold" aria-hidden />}
        {offline ? "No signal · Readings wait on the device" : "Online · 2 cameras"}
      </p>
    </article>
  );
}

function Controls() {
  const s = useSim();
  const stations = line.data.stations;
  const toTime = (v: string) => { const [h, m] = v.split(":").map(Number); return h * 3600 + m * 60; };
  return (
    <aside className="grid gap-4 lg:sticky lg:top-20" aria-label="Scenario">
      <section className="card grid gap-3 p-4">
        <h2 className="text-base font-bold text-ink">Scenario</h2>
        <div className="flex flex-wrap gap-2">
          {SCENARIOS.map((sc) => <Pill key={sc.key} on={s.scenarioKey === sc.key} onClick={() => s.chooseScenario(sc.key)}>{sc.label}</Pill>)}
        </div>
        <p className="text-xs text-ink-3">{SCENARIOS.find((x) => x.key === s.scenarioKey)?.hint}</p>
      </section>
      <section className="card grid gap-3 p-4">
        <h2 className="text-base font-bold text-ink">Train</h2>
        <label className="grid gap-1 text-sm font-semibold text-ink">Station
          <select value={s.stationId} onChange={(e) => s.restart({ stationId: e.target.value })}
            className="rounded-sign border border-rule bg-surface px-3 py-2 font-medium">
            {stations.map((st) => <option key={st.id} value={st.id}>{st.name} ({st.id})</option>)}
          </select>
        </label>
        <div className="grid gap-1 text-sm font-semibold text-ink">Direction
          <div className="flex flex-wrap gap-2" role="group" aria-label="Direction">
            {([0, 1] as const).map((d) => <Pill key={d} on={s.direction === d} onClick={() => s.restart({ direction: d })}>To {line.dir(d).headsign}</Pill>)}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="grid gap-1 text-sm font-semibold text-ink">Day
            <select value={s.weekday} onChange={(e) => s.restart({ weekday: Number(e.target.value) })}
              className="rounded-sign border border-rule bg-surface px-3 py-2 font-medium">
              {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
          </label>
          <label className="grid gap-1 text-sm font-semibold text-ink">Time
            <input type="time" value={clock(s.timeS, false)} step={60} onChange={(e) => e.target.value && s.restart({ timeS: toTime(e.target.value) })}
              className="num rounded-sign border border-rule bg-surface px-3 py-2 font-medium" />
          </label>
        </div>
        <button type="button" onClick={() => s.restart()} className="pressable inline-flex items-center justify-center gap-2 rounded-full border border-rule px-4 py-2 text-sm font-semibold text-ink-2 hover:bg-surface-2">
          <ArrowClockwise size={16} weight="bold" aria-hidden />Start the trip again
        </button>
      </section>
      <section className="card grid gap-3 p-4">
        <h2 className="text-base font-bold text-ink">Network</h2>
        <Switch on={s.offline} onChange={s.setOffline} label="No signal (tunnel)"
          hint="Readings wait on the device and are dropped after 2 minutes" />
        <p className="text-xs text-ink-3">{s.send.on ? "Sending to AERA" : "Readings are logged here, not sent"} ·{" "}
          <Link href="/settings" className="font-semibold text-accent">Change</Link></p>
        <Link href="/camera" className="pressable inline-flex items-center justify-center gap-2 rounded-full bg-ink px-4 py-2.5 text-sm font-semibold text-paper">
          <VideoCamera size={17} weight="fill" aria-hidden />Open the cameras
        </Link>
      </section>
    </aside>
  );
}
