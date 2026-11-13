/** checkCarCount must agree with AERA's JSON Schema (plus AERA's "one measure and a train" rule). */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { checkCarCount } from "./contract.ts";

const schema = JSON.parse(readFileSync(new URL("../../../contracts/car-count.schema.json", import.meta.url), "utf8"));
const ajv = new Ajv2020({ strict: false });
addFormats.default(ajv);
const validate = ajv.compile(schema);

/** AERA's pydantic model_validator, on top of the schema. */
const aeraRule = (m: Record<string, unknown>) =>
  (m.count == null) !== (m.load_pct == null) && (m.trip_id != null || (m.station_id != null && m.direction_id != null));

const good = { v: 1, seq: 4, ts: "2026-11-13T08:00:00.000Z", method: "camera", car: 2, count: 97, confidence: 0.84, trip_id: "KJ-0715-1" };

const cases: Record<string, unknown>[] = [
  good,
  { ...good, trip_id: undefined, station_id: "KJ15", direction_id: 1 },
  { ...good, trip_id: undefined, station_id: "KJ15" },
  { ...good, count: undefined, load_pct: 48.5 },
  { ...good, load_pct: 48.5 },
  { ...good, count: undefined },
  { ...good, count: 601 },
  { ...good, count: 600 },
  { ...good, count: 12.5 },
  { ...good, car: 0 },
  { ...good, car: 5 },
  { ...good, confidence: 1.2 },
  { ...good, confidence: 0 },
  { ...good, v: 2 },
  { ...good, seq: -1 },
  { ...good, method: "lidar" },
  { ...good, ts: "yesterday" },
  { ...good, extra: true },
  { ...good, direction_id: 2, trip_id: undefined, station_id: "KJ1" },
  { ...good, trip_id: "" },
  { ...good, station_id: "X".repeat(17) },
].map((c) => Object.fromEntries(Object.entries(c).filter(([, v]) => v !== undefined)));

test("hand validator agrees with AERA's schema on every case", () => {
  for (const c of cases) {
    const aera = validate(c) && aeraRule(c);
    const ours = checkCarCount(c).length === 0;
    assert.equal(ours, aera, `disagree on ${JSON.stringify(c)}: ${checkCarCount(c).join("; ")}`);
  }
});

test("our schema copy is AERA's file (when AERA is next to this repo)", (t) => {
  const aeraPath = new URL("../../../../aera-advanced ecosystem-of-rail-analytics/packages/contracts/car-count.schema.json", import.meta.url);
  let theirs: unknown;
  try {
    theirs = JSON.parse(readFileSync(aeraPath, "utf8"));
  } catch {
    t.skip("AERA repo not found");
    return;
  }
  assert.deepEqual(schema, theirs);
});
