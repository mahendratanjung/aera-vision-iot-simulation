"use client";
/**
 * App state. The simulation engine lives outside React; `tick` changes after every step so screens
 * re-render. The AERA token is kept in memory only: it is never written to storage.
 */
import { create } from "zustand";
import { Engine, SCENARIOS, type BankFrame, type LogEntry } from "@/lib/sim/engine";
import { Line, type LineData } from "@/lib/sim/line";
import type { Post } from "@/lib/device/sender";
import type { Band } from "@/lib/device/confidence";
import kj from "@/data/kj.json";

export const line = new Line(kj as unknown as LineData);

export interface Bank {
  model: { key: string; label: string; kind: string; threshold: number | null };
  attribution: string;
  frames: BankFrame[];
}

export type Speed = 1 | 5 | 20;

export interface SendSettings {
  on: boolean;
  url: string;
  token: string;
  tripId: string | null;
  status: "idle" | "checking" | "ok" | "error";
  message: string;
}

interface State {
  engine: Engine | null;
  bank: Bank | null;
  bands: Band[];
  bankError: string | null;
  tick: number;
  running: boolean;
  speed: Speed;
  scenarioKey: string;
  stationId: string;
  direction: 0 | 1;
  weekday: number;
  timeS: number;
  selectedCar: number;
  offline: boolean;
  send: SendSettings;
  logs: LogEntry[];
  setBank: (bank: Bank | null, bands: Band[], error?: string) => void;
  restart: (patch?: Partial<Pick<State, "scenarioKey" | "stationId" | "direction" | "weekday" | "timeS">>) => void;
  chooseScenario: (key: string) => void;
  step: (realSeconds: number) => void;
  setRunning: (on: boolean) => void;
  setSpeed: (s: Speed) => void;
  setOffline: (on: boolean) => void;
  selectCar: (car: number) => void;
  setSend: (patch: Partial<SendSettings>) => void;
}

const MAX_LOGS = 300;

/** Unix seconds of today's midnight in Kuala Lumpur (UTC+8). */
export function klMidnight(nowMs = Date.now()): number {
  const kl = nowMs / 1000 + 8 * 3600;
  return Math.floor(kl / 86400) * 86400 - 8 * 3600;
}

export function klNow(nowMs = Date.now()): { weekday: number; timeS: number } {
  const kl = new Date(nowMs + 8 * 3600 * 1000);
  return { weekday: (kl.getUTCDay() + 6) % 7, timeS: kl.getUTCHours() * 3600 + kl.getUTCMinutes() * 60 + kl.getUTCSeconds() };
}

const logPost: Post = async () => ({ status: 202, body: { accepted: true } });

/** Posts through the app's own forwarder (AERA only allows its own web origin). */
export function aeraPost(target: string): Post {
  return async (url, body, headers) => {
    const res = await fetch("/api/forward", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target, path: new URL(url).pathname, body, token: headers["X-Ingest-Token"] ?? "" }),
    });
    if (res.status === 502) throw new Error("AERA unreachable");
    const json = await res.json().catch(() => ({}));
    return { status: json.status ?? res.status, body: json.body };
  };
}

const first = SCENARIOS[1];

export const useSim = create<State>((set, get) => {
  const build = (s: State) => {
    if (!s.bank) return null;
    const scenario = SCENARIOS.find((x) => x.key === s.scenarioKey) ?? first;
    const live = s.send.on;
    return new Engine({
      line, bank: s.bank.frames, bands: s.bands, scenario,
      stationId: s.stationId, direction: s.direction, weekday: s.weekday, timeS: s.timeS,
      tripId: live ? s.send.tripId : null,
      post: live ? aeraPost(s.send.url) : logPost, mode: live ? "aera" : "log", apiUrl: live ? s.send.url : "http://aera.local",
      token: s.send.token, dateBase: klMidnight(),
      onLog: (e) => set((st) => ({ logs: [e, ...st.logs].slice(0, MAX_LOGS) })),
    });
  };
  return {
    engine: null, bank: null, bands: [], bankError: null, tick: 0, running: false, speed: 5,
    scenarioKey: first.key, stationId: first.stationId, direction: first.direction, weekday: first.weekday, timeS: first.timeS,
    selectedCar: 1, offline: false, logs: [],
    send: { on: false, url: "http://localhost:8000", token: "", tripId: null, status: "idle", message: "" },
    setBank: (bank, bands, error) => {
      set({ bank, bands, bankError: error ?? null });
      const e = build(get());
      set({ engine: e, tick: get().tick + 1 });
    },
    restart: (patch = {}) => {
      set({ ...patch, logs: [] });
      const e = build(get());
      if (e && get().offline) e.setOffline(true);
      set({ engine: e, tick: get().tick + 1 });
    },
    chooseScenario: (key) => {
      const sc = SCENARIOS.find((x) => x.key === key) ?? first;
      get().restart({ scenarioKey: sc.key, stationId: sc.stationId, direction: sc.direction, weekday: sc.weekday, timeS: sc.timeS });
    },
    step: (realSeconds) => {
      const { engine, speed, running } = get();
      if (!engine || !running) return;
      engine.advance(engine.t + realSeconds * speed);
      set({ tick: get().tick + 1, running: engine.train.phase !== "ended" });
    },
    setRunning: (on) => set({ running: on }),
    setSpeed: (speed) => set({ speed }),
    setOffline: (on) => {
      get().engine?.setOffline(on);
      set({ offline: on, tick: get().tick + 1 });
    },
    selectCar: (car) => set({ selectedCar: car }),
    setSend: (patch) => set((s) => ({ send: { ...s.send, ...patch } })),
  };
});
