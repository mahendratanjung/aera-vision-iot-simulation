/**
 * The simulated train: four cars, two cameras per car, one counting device per car.
 *
 * Honesty rule: every count on screen is real. The scenario (AERA's typical crowding for that
 * stretch and hour) only decides which held-out test photo each camera "sees". The count the
 * device works with is the chosen model's actual output on that photo (precomputed by
 * model/aera_vision/demo.py), and the photo's labelled count is shown next to it.
 *
 * Per car the device does what AERA's edge code does: per-frame count -> CountSmoother -> Sender
 * (seq, queue, 2-minute drop). Time is the simulation clock, in seconds.
 */
import { CARS, CAMERAS_PER_CAR, FPS, TRAIN_CAPACITY } from "./constants.ts";
import { type Line, type Surge } from "./line.ts";
import { CountSmoother } from "../device/smoother.ts";
import { Sender, type Post, type SendEvent } from "../device/sender.ts";
import { carConfidence, type Band } from "../device/confidence.ts";
import type { CarCount } from "../device/contract.ts";

export interface BankFrame {
  id: string;
  file: string;
  w: number;
  h: number;
  truth: number; // labelled people (CrowdHuman), after the do-not-count rule
  model: number; // the chosen model's count on this photo
  boxes?: number[][]; // [x1, y1, x2, y2] in the stored image's pixels, detections kept by the model
  points?: number[][]; // [x, y, weight] for a density model
}

export interface Scenario {
  key: string;
  label: string;
  hint: string;
  weekday: number; // 0 = Monday
  timeS: number; // seconds after midnight
  stationId: string;
  direction: 0 | 1;
  multiplier: number;
  carShares: [number, number, number, number]; // relative fullness, front car first
  surge?: Surge;
}

export const SCENARIOS: Scenario[] = [
  { key: "quiet", label: "Quiet morning", hint: "Sunday 07:30, plenty of seats", weekday: 6, timeS: 7.5 * 3600,
    stationId: "KJ24", direction: 0, multiplier: 1, carShares: [0.9, 1.05, 1.05, 1] },
  { key: "rush", label: "Rush hour", hint: "Weekday 08:00 towards the city", weekday: 1, timeS: 8 * 3600,
    stationId: "KJ21", direction: 0, multiplier: 1, carShares: [0.92, 1.06, 1.06, 0.96] },
  { key: "event", label: "Event surge", hint: "Weekday 22:30, a crowd boarding at KLCC", weekday: 4, timeS: 22.5 * 3600,
    stationId: "KJ10", direction: 1, multiplier: 1, carShares: [0.95, 1.05, 1.05, 0.95], surge: { stationId: "KJ10", multiplier: 6 } },
  { key: "uneven", label: "Packed car next to an empty one", hint: "Weekday 18:00, everyone in the front car",
    weekday: 2, timeS: 18 * 3600, stationId: "KJ13", direction: 1, multiplier: 1, carShares: [1.75, 0.35, 0.95, 0.95] },
];

export interface TrainState {
  phase: "dwelling" | "moving" | "ended";
  stopIndex: number; // last stop reached
  nextIndex: number | null;
  progress: number; // 0..1 between stops when moving
  loadPct: number; // AERA's typical load for the current leg, with the scenario applied
}

export interface View { frame: BankFrame; target: number }

export interface CarState {
  car: number;
  deviceId: string;
  views: View[];
  truth: number; // labelled people in both photos
  model: number; // the model's count for both photos (what the device sees this frame)
  lastValue: number | null; // last smoothed value sent
  lastReading: CarCount | null;
  lastReadingAt: number | null;
  confidence: number;
  sdPoints: number;
  queue: number;
  sent: number;
  dropped: number;
  rejected: number;
}

export interface HistoryPoint { t: number; truth: number; model: number; sent: number | null }

export interface LogEntry extends SendEvent { car: number; deviceId: string; msg?: CarCount; mode: "log" | "aera" }

export interface EngineOptions {
  line: Line;
  bank: BankFrame[];
  bands: Band[];
  scenario: Scenario;
  stationId?: string;
  direction?: 0 | 1;
  weekday?: number;
  timeS?: number;
  tripId?: string | null;
  post: Post;
  mode: "log" | "aera";
  apiUrl: string;
  token: string;
  dateBase?: number; // unix seconds of midnight for the simulated day (KL), for timestamps
  onLog?: (e: LogEntry) => void;
}

/** Tiny deterministic hash, so the same scene always shows the same photos. */
function hash(...xs: (number | string)[]): number {
  let h = 2166136261;
  for (const ch of xs.join("|")) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967296;
}

/** The bank photo for a view with roughly `target` people: one of the closest, varied by `seed`. */
export function pickFrame(bank: BankFrame[], target: number, seed: number): BankFrame {
  const sorted = [...bank].sort((a, b) => Math.abs(a.truth - target) - Math.abs(b.truth - target) || a.id.localeCompare(b.id));
  const tol = Math.max(2, target * 0.1);
  const close = sorted.filter((f) => Math.abs(f.truth - target) <= tol).slice(0, 4);
  const pool = close.length ? close : sorted.slice(0, 1);
  return pool[Math.floor(seed * pool.length) % pool.length];
}

export class Engine {
  readonly opts: EngineOptions;
  readonly line: Line;
  readonly direction: 0 | 1;
  readonly weekday: number;
  readonly startStation: string;
  readonly tripStart: number; // sim seconds of day the train left its origin
  t: number;
  tripId: string | null;
  train: TrainState;
  cars: CarState[];
  history: HistoryPoint[][];
  offline = false;
  private smoothers: CountSmoother[];
  private senders: Sender[];
  private sceneKey = "";
  private nextFrameAt: number;

  constructor(opts: EngineOptions) {
    this.opts = opts;
    this.line = opts.line;
    const sc = opts.scenario;
    this.direction = opts.direction ?? sc.direction;
    this.weekday = opts.weekday ?? sc.weekday;
    this.startStation = opts.stationId ?? sc.stationId;
    const t0 = opts.timeS ?? sc.timeS;
    const k = Math.max(0, this.line.stopIndex(this.direction, this.startStation));
    this.tripStart = t0 - this.line.dir(this.direction).stops[k].arr;
    this.t = t0;
    this.nextFrameAt = t0;
    this.tripId = opts.tripId ?? null;
    this.smoothers = Array.from({ length: CARS }, () => new CountSmoother());
    this.senders = Array.from({ length: CARS }, (_, i) => new Sender(opts.apiUrl, deviceId(i + 1), opts.token, this.postGuarded(),
      () => this.clock(), 500, 120, (e) => this.log(i + 1, e)));
    this.history = Array.from({ length: CARS }, () => []);
    this.train = this.trainAt(t0);
    this.cars = Array.from({ length: CARS }, (_, i) => this.emptyCar(i + 1));
    this.updateScene(true);
  }

  /** Unix seconds for sim time t, for reading timestamps and the 2-minute rule. */
  clock(): number {
    return (this.opts.dateBase ?? 0) + this.t;
  }

  private postGuarded(): Post {
    return async (url, body, headers) => {
      if (this.offline) throw new Error("no signal");
      return this.opts.post(url, body, headers);
    };
  }

  private lastMsgs = new Map<string, CarCount>();

  private log(car: number, e: SendEvent) {
    const msg = this.lastMsgs.get(`${car}:${e.seq}`);
    this.opts.onLog?.({ ...e, car, deviceId: deviceId(car), msg, mode: this.opts.mode });
    const c = this.cars?.[car - 1];
    const s = this.senders[car - 1];
    if (c) Object.assign(c, { queue: s.queue.length, sent: s.sent, dropped: s.dropped, rejected: s.rejected });
  }

  private emptyCar(car: number): CarState {
    return { car, deviceId: deviceId(car), views: [], truth: 0, model: 0, lastValue: null, lastReading: null, lastReadingAt: null,
      confidence: 1, sdPoints: 0, queue: 0, sent: 0, dropped: 0, rejected: 0 };
  }

  /** Where the train is at sim time t, and AERA's typical load for that leg. */
  trainAt(t: number): TrainState {
    const stops = this.line.dir(this.direction).stops;
    const e = t - this.tripStart;
    const sc = this.opts.scenario;
    const load = (k: number) => this.line.legLoad(this.direction, k, this.weekday, t, sc.surge) * sc.multiplier;
    const last = stops.length - 1;
    if (e >= stops[last].arr) return { phase: "ended", stopIndex: last, nextIndex: null, progress: 1, loadPct: 0 };
    for (let i = 0; i < last; i++) {
      if (e < stops[i].arr) continue;
      if (e <= stops[i].dep) {
        const f = (e - stops[i].arr) / Math.max(1, stops[i].dep - stops[i].arr);
        const before = i > 0 ? load(i - 1) : 0;
        return { phase: "dwelling", stopIndex: i, nextIndex: i + 1, progress: 0, loadPct: before + (load(i) - before) * f };
      }
      if (e < stops[i + 1].arr) {
        return { phase: "moving", stopIndex: i, nextIndex: i + 1,
          progress: (e - stops[i].dep) / Math.max(1, stops[i + 1].arr - stops[i].dep), loadPct: load(i) };
      }
    }
    return { phase: "dwelling", stopIndex: 0, nextIndex: 1, progress: 0, loadPct: 0 };
  }

  /** Target people per car from the train's load and the scenario's car shares. */
  carTargets(): number[] {
    const shares = this.opts.scenario.carShares;
    const mean = shares.reduce((a, b) => a + b, 0) / CARS;
    return shares.map((s) => (this.train.loadPct / 100) * TRAIN_CAPACITY / CARS * (s / mean));
  }

  /** New photos when the doors open, halfway through the stop, and when the train leaves. */
  private updateScene(force = false) {
    const tr = this.train;
    const stop = this.line.dir(this.direction).stops[tr.stopIndex];
    const e = this.t - this.tripStart;
    const stage = tr.phase !== "dwelling" ? tr.phase : (e - stop.arr) * 2 < stop.dep - stop.arr ? "doors-open" : "boarding";
    const key = `${tr.stopIndex}:${stage}`;
    if (!force && key === this.sceneKey) return;
    this.sceneKey = key;
    const targets = this.carTargets();
    this.cars.forEach((c, i) => {
      const imbalance = 0.06 * (hash("half", i) - 0.5) * 2;
      const halves = [targets[i] * (0.5 + imbalance / 2), targets[i] * (0.5 - imbalance / 2)];
      c.views = halves.slice(0, CAMERAS_PER_CAR).map((target, h) => ({
        target, frame: pickFrame(this.opts.bank, target, hash(this.opts.scenario.key, i, h, key)) }));
      c.truth = c.views.reduce((a, v) => a + v.frame.truth, 0);
      c.model = c.views.reduce((a, v) => a + v.frame.model, 0);
      const cc = carConfidence(c.views[0].frame.model, c.views[1].frame.model, this.opts.bands);
      c.confidence = cc.confidence;
      c.sdPoints = cc.sdPoints;
    });
  }

  /** Name of the train in readings: trip id if known, else the last station and direction. */
  private trainRef(): Pick<CarCount, "trip_id" | "station_id" | "direction_id"> {
    if (this.tripId) return { trip_id: this.tripId };
    const stops = this.line.dir(this.direction).stops;
    return { station_id: stops[this.train.stopIndex].station_id, direction_id: this.direction };
  }

  /** Advance the simulation to sim time t, one camera frame (1 / FPS s) at a time. */
  advance(toT: number) {
    while (this.nextFrameAt <= toT) {
      this.t = this.nextFrameAt;
      this.nextFrameAt += 1 / FPS;
      this.train = this.trainAt(this.t);
      if (this.train.phase === "ended") break;
      this.updateScene();
      this.cars.forEach((c, i) => {
        const v = this.smoothers[i].add(c.model, this.t);
        if (v === null) return;
        const s = this.senders[i];
        const msg = s.reading({ car: c.car, method: "camera", count: Math.min(600, Math.max(0, v)), confidence: c.confidence,
          ...this.trainRef() });
        this.lastMsgs.set(`${c.car}:${msg.seq}`, msg);
        if (this.lastMsgs.size > 400) this.lastMsgs.delete(this.lastMsgs.keys().next().value!);
        c.lastValue = v;
        c.lastReading = msg;
        c.lastReadingAt = this.t;
        void s.send(msg).then(() => Object.assign(c, { queue: s.queue.length }));
        c.queue = s.queue.length;
      });
      if (Math.round(this.t * FPS) % (10 * FPS) === 0) {
        this.cars.forEach((c, i) => {
          const h = this.history[i];
          h.push({ t: this.t, truth: c.truth, model: c.model, sent: c.lastValue });
          if (h.length > 360) h.shift();
        });
      }
    }
    this.t = Math.max(this.t, Math.min(toT, this.nextFrameAt));
  }

  setOffline(off: boolean) {
    this.offline = off;
    if (!off) this.senders.forEach((s) => void s.flush());
  }

  setConnection(post: Post, apiUrl: string, token: string, mode: "log" | "aera") {
    this.opts.post = post;
    this.opts.mode = mode;
    this.opts.apiUrl = apiUrl;
    this.senders.forEach((s) => s.setTarget(apiUrl, token));
  }

  stationName(k: number): string {
    return this.line.name(this.line.dir(this.direction).stops[k].station_id);
  }
}

export function deviceId(car: number): string {
  return `cam-t07-c${car}`;
}
