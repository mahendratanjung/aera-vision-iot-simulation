/** The device logic, with the same cases as AERA's edge/tests/test_edge.py plus the queue limits. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { Sender, type Post } from "./sender.ts";
import { CountSmoother, median, roundHalfEven } from "./smoother.ts";
import { carConfidence } from "./confidence.ts";

function fakeNet() {
  const net = { status: 202 as number | null, sent: [] as Record<string, unknown>[], headers: {} as Record<string, string> };
  const post: Post = async (_url, body, headers) => {
    if (net.status === null) throw new Error("network down");
    net.sent.push(JSON.parse(body));
    net.headers = headers;
    return { status: net.status, body: { accepted: true } };
  };
  return { net, post };
}

function make(post: Post, now: { t: number }, queueMax = 500) {
  return new Sender("http://twin:8000/", "cam-1", "secret", post, () => now.t, queueMax);
}

test("readings follow the contract", async () => {
  const { net, post } = fakeNet();
  const now = { t: 1_800_000_000 };
  const s = make(post, now);
  assert.equal(s.url, "http://twin:8000/api/ingest/train/cam-1");
  const msg = s.reading({ car: 2, method: "camera", count: 40, station_id: "KJ10", direction_id: 1 });
  assert.equal(msg.v, 1);
  assert.equal(msg.seq, 1);
  assert.equal(msg.ts, "2027-01-15T08:00:00.000Z");
  assert.ok(!("trip_id" in msg));
  assert.equal(await s.send(msg), true);
  assert.equal(net.headers["X-Ingest-Token"], "secret");
});

test("offline readings wait in order and stale ones are dropped", async () => {
  const { net, post } = fakeNet();
  const now = { t: 1000 };
  const s = make(post, now);
  net.status = null; // in the tunnel
  assert.equal(await s.send(s.reading({ car: 1, method: "door_counter", count: 10, trip_id: "T" })), false);
  now.t += 30;
  assert.equal(await s.send(s.reading({ car: 1, method: "door_counter", count: 12, trip_id: "T" })), false);
  net.status = 202;
  now.t += 5;
  assert.equal(await s.flush(), true);
  assert.deepEqual(net.sent.map((m) => m.count), [10, 12]);
  net.status = null;
  net.sent = [];
  await s.send(s.reading({ car: 1, method: "door_counter", count: 20, trip_id: "T" }));
  now.t += 300; // five minutes later the reading is no use to riders
  net.status = 202;
  assert.equal(await s.flush(), true);
  assert.deepEqual(net.sent, []);
  assert.equal(s.dropped, 1);
});

test("a reading exactly 2 minutes old is still sent; just over is dropped", async () => {
  const { net, post } = fakeNet();
  const now = { t: 0 };
  const s = make(post, now);
  net.status = null;
  await s.send(s.reading({ car: 1, method: "camera", count: 1, trip_id: "T" }));
  now.t = 120;
  net.status = 202;
  await s.flush();
  assert.equal(net.sent.length, 1);
  net.status = null;
  await s.send(s.reading({ car: 1, method: "camera", count: 2, trip_id: "T" }));
  now.t = 240.5;
  net.status = 202;
  await s.flush();
  assert.equal(net.sent.length, 1);
});

test("rejected readings are not retried; server errors and 429 are", async () => {
  const { net, post } = fakeNet();
  const now = { t: 0 };
  const s = make(post, now);
  net.status = 401;
  assert.equal(await s.send(s.reading({ car: 1, method: "camera", count: 1, trip_id: "T" })), true);
  assert.equal(s.rejected, 1);
  for (const status of [503, 429]) {
    net.status = status;
    assert.equal(await s.send(s.reading({ car: 1, method: "camera", count: 2, trip_id: "T" })), false);
    assert.equal(s.queue.length, 1);
    net.status = 202;
    assert.equal(await s.flush(), true);
    assert.equal(s.queue.length, 0);
  }
});

test("the queue keeps at most queueMax readings, oldest pushed out", async () => {
  const { net, post } = fakeNet();
  const now = { t: 0 };
  const s = make(post, now, 3);
  net.status = null;
  for (let i = 1; i <= 5; i++) await s.send(s.reading({ car: 1, method: "camera", count: i, trip_id: "T" }));
  assert.deepEqual(s.queue.map((q) => q.msg.count), [3, 4, 5]);
  assert.equal(s.dropped, 2);
  net.status = 202;
  await s.flush();
  assert.deepEqual(net.sent.map((m) => m.seq), [3, 4, 5]);
});

test("smoother reports steady medians (AERA's case)", () => {
  const sm = new CountSmoother(5, 3, 30);
  const out = [10, 30, 11, 9, 10, 12, 10, 11].map((c, t) => sm.add(c, t));
  assert.deepEqual(out.slice(0, 4), [null, null, null, null]);
  assert.equal(out[4], 10); // the 30 (a double detection) is ignored
  assert.ok(out.slice(5).every((v) => v === null)); // small wobble is not resent
  assert.equal(sm.add(25, 40), 11); // nothing new for 30 s: send again anyway
});

test("smoother defaults match AERA (window 9, change 3, every 30 s)", () => {
  const sm = new CountSmoother();
  assert.deepEqual([sm.window, sm.minChange, sm.everyS], [9, 3, 30]);
  for (let t = 0; t < 8; t++) assert.equal(sm.add(50, t), null);
  assert.equal(sm.add(50, 8), 50);
});

test("median and Python-style rounding", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.deepEqual([0.5, 1.5, 2.5, 2.4, 2.6].map(roundHalfEven), [0, 2, 2, 2, 3]);
});

test("confidence inverts AERA's error rule", () => {
  const bands = [1, 2, 3, 4, 5].map((level) => ({ level, rmse_used: [2, 4, 10, 20, 30][level - 1] }));
  // both views Quiet (error 2 people each): sqrt(8) / 200 * 100 = 1.41 points, 8 / 1.41 > 1
  assert.equal(carConfidence(5, 10, bands).confidence, 1);
  // both views Packed (30 each): sqrt(1800) / 2 = 21.2 points, 8 / 21.2 = 0.377
  const packed = carConfidence(98, 99, bands);
  assert.ok(Math.abs(packed.sdPoints - Math.sqrt(1800) / 2) < 1e-9);
  assert.equal(packed.confidence, 0.38);
  // AERA never trusts below 0.3
  assert.equal(carConfidence(98, 99, bands.map((b) => ({ ...b, rmse_used: 500 }))).confidence, 0.3);
});
