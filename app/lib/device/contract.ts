/**
 * The message an in-train sensor sends to AERA: `CarCount`, as defined by AERA's
 * packages/contracts/car-count.schema.json (copy in ../../contracts). contract.test.ts checks
 * `checkCarCount` agrees with the JSON Schema itself, including AERA's extra rule that exactly one
 * of count / load_pct is sent and the train is named by trip_id or by station_id + direction_id.
 */

export interface CarCount {
  v: 1;
  seq: number;
  ts: string;
  method: "camera" | "door_counter" | "load_weigh";
  car: number;
  count?: number;
  load_pct?: number;
  confidence?: number;
  trip_id?: string;
  station_id?: string;
  direction_id?: 0 | 1;
}

const KEYS = new Set(["v", "seq", "ts", "method", "car", "count", "load_pct", "confidence", "trip_id", "station_id", "direction_id"]);
const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const int = (x: unknown) => typeof x === "number" && Number.isInteger(x);
const num = (x: unknown) => typeof x === "number" && Number.isFinite(x);
const str = (x: unknown, max: number) => typeof x === "string" && x.length >= 1 && x.length <= max;

/** Problems with a reading, in plain words. Empty when AERA would accept its shape. */
export function checkCarCount(m: Record<string, unknown>): string[] {
  const p: string[] = [];
  for (const k of Object.keys(m)) if (!KEYS.has(k)) p.push(`Unknown field ${k}`);
  if (m.v !== 1) p.push("v must be 1");
  if (!int(m.seq) || (m.seq as number) < 0) p.push("seq must be a whole number, 0 or more");
  if (typeof m.ts !== "string" || !ISO.test(m.ts)) p.push("ts must be a date and time");
  if (!["camera", "door_counter", "load_weigh"].includes(m.method as string)) p.push("method is not one AERA knows");
  if (!int(m.car) || (m.car as number) < 1 || (m.car as number) > 4) p.push("car must be 1 to 4");
  if (m.count != null && (!int(m.count) || (m.count as number) < 0 || (m.count as number) > 600)) p.push("count must be 0 to 600");
  if (m.load_pct != null && (!num(m.load_pct) || (m.load_pct as number) < 0 || (m.load_pct as number) > 250)) p.push("load_pct must be 0 to 250");
  if (m.confidence != null && (!num(m.confidence) || (m.confidence as number) < 0 || (m.confidence as number) > 1)) p.push("confidence must be 0 to 1");
  if (m.trip_id != null && !str(m.trip_id, 64)) p.push("trip_id must be 1 to 64 characters");
  if (m.station_id != null && !str(m.station_id, 16)) p.push("station_id must be 1 to 16 characters");
  if (m.direction_id != null && m.direction_id !== 0 && m.direction_id !== 1) p.push("direction_id must be 0 or 1");
  if ((m.count == null) === (m.load_pct == null)) p.push("Send exactly one of count or load_pct");
  if (m.trip_id == null && (m.station_id == null || m.direction_id == null)) p.push("Name the train with trip_id, or station_id and direction_id");
  return p;
}

export const DEVICE_PATTERN = /^[A-Za-z0-9_-]{1,64}$/; // AERA's crowdsource/riders.py DEVICE_PATTERN
export const MAX_BODY_BYTES = 4096; // AERA's routes.py MAX_SENSOR_BODY
