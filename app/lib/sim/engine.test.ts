/** The simulated train, on the real line data with a small synthetic photo bank (test fixture only). */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Engine, pickFrame, SCENARIOS, type BankFrame, type LogEntry } from "./engine.ts";
import { Line, type LineData } from "./line.ts";
import { checkCarCount } from "../device/contract.ts";
import type { Post } from "../device/sender.ts";

const line = new Line(JSON.parse(readFileSync(new URL("../../data/kj.json", import.meta.url), "utf8")) as LineData);
// fixture: one photo for every 2 people from 0 to 160; the "model" undercounts big crowds by 10 %
const bank: BankFrame[] = Array.from({ length: 81 }, (_, i) => ({
  id: `f${i}`, file: `f${i}.jpg`, w: 10, h: 10, truth: i * 2, model: Math.round(i * 2 * (i > 40 ? 0.9 : 1)),
}));
const bands = [1, 2, 3, 4, 5].map((level) => ({ level, rmse_used: [1.5, 4, 8, 14, 22][level - 1] }));
const ok: Post = async () => ({ status: 202, body: { accepted: true } });

function engine(key: string, post: Post = ok, onLog?: (e: LogEntry) => void) {
  const scenario = SCENARIOS.find((s) => s.key === key)!;
  return new Engine({ line, bank, bands, scenario, post, mode: "log", apiUrl: "http://x", token: "", onLog });
}

test("the train is at the chosen station at the chosen time", () => {
  const e = engine("rush");
  assert.equal(e.train.phase, "dwelling");
  assert.equal(line.dir(0).stops[e.train.stopIndex].station_id, "KJ21");
  e.advance(e.t + 600);
  assert.ok(e.train.stopIndex > line.stopIndex(0, "KJ21"), "moved on towards Gombak");
});

test("rush hour is fuller than a quiet Sunday morning, and the event adds a crowd at KLCC", () => {
  const quiet = engine("quiet").train.loadPct;
  const rush = engine("rush").train.loadPct;
  assert.ok(rush > quiet * 2, `rush ${rush} vs quiet ${quiet}`);
  const ev = engine("event");
  ev.advance(ev.t + 40); // leave KLCC
  const plain = line.legLoad(1, line.stopIndex(1, "KJ10"), 4, ev.t);
  assert.ok(ev.train.loadPct > plain * 1.5, `surge ${ev.train.loadPct} vs ${plain}`);
});

test("uneven scenario: the front car is far fuller than car 2", () => {
  const e = engine("uneven");
  const t = e.carTargets();
  assert.ok(t[0] > 4 * t[1]);
  assert.ok(e.cars[0].truth > e.cars[1].truth);
});

test("pickFrame takes one of the closest photos", () => {
  for (const target of [0, 7, 51, 133]) {
    const f = pickFrame(bank, target, 0.37);
    assert.ok(Math.abs(f.truth - target) <= Math.max(2, target * 0.1), `${target} -> ${f.truth}`);
  }
  assert.equal(pickFrame(bank, 999, 0.5).truth, 160); // beyond the bank: the fullest photo
});

test("every reading is a valid CarCount with rising seq and an AERA-ready confidence", async () => {
  const logs: LogEntry[] = [];
  const e = engine("rush", ok, (x) => logs.push(x));
  e.advance(e.t + 900);
  await new Promise((r) => setTimeout(r, 10));
  const sent = logs.filter((l) => l.kind === "sent");
  assert.ok(sent.length >= 4 * 5, `only ${sent.length} readings in 15 minutes`);
  for (const car of [1, 2, 3, 4]) {
    const seqs = sent.filter((l) => l.car === car).map((l) => l.msg!.seq);
    assert.deepEqual(seqs, [...seqs].sort((a, b) => a - b));
  }
  for (const l of sent) {
    assert.deepEqual(checkCarCount(l.msg as unknown as Record<string, unknown>), []);
    assert.ok(l.msg!.confidence! >= 0.3 && l.msg!.confidence! <= 1);
    assert.equal(l.msg!.method, "camera");
  }
});

test("in a tunnel readings queue; after 2 minutes they are dropped, newer ones are sent", async () => {
  const logs: LogEntry[] = [];
  const e = engine("rush", ok, (x) => logs.push(x));
  e.advance(e.t + 60);
  e.setOffline(true);
  e.advance(e.t + 300);
  assert.ok(e.cars.some((c) => c.queue > 0), "readings wait while offline");
  e.setOffline(false);
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(logs.some((l) => l.kind === "dropped"), "old readings were dropped");
  assert.ok(e.cars.every((c) => c.queue === 0));
});
