/**
 * Sends in-train readings to AERA, with the same rules as AERA's edge/aera_edge/sender.py:
 *
 * - `seq` goes up by one for every reading the device makes.
 * - Readings wait in a queue (at most `queueMax`, oldest pushed out) and are sent in order.
 * - A reading older than `maxAgeS` (2 minutes) when its turn comes is dropped: old crowding
 *   misleads riders.
 * - Network failure, 5xx or 429: stop and keep the queue for the next try.
 * - Any other 4xx (bad token, bad reading): count it as rejected and move on; retrying will not help.
 */
import type { CarCount } from "./contract";

/** Posts one JSON body; resolves to the HTTP status, rejects when the network is down. */
export type Post = (url: string, body: string, headers: Record<string, string>) => Promise<{ status: number; body?: unknown }>;

export interface SendEvent {
  at: number; // sender clock, seconds
  seq: number;
  kind: "sent" | "rejected" | "dropped" | "offline" | "retry";
  status?: number;
  detail?: string;
}

export class Sender {
  url: string;
  queue: { at: number; msg: CarCount }[] = [];
  seq = 0;
  rejected = 0;
  dropped = 0;
  sent = 0;
  private flushing: Promise<boolean> | null = null;
  readonly deviceId: string;
  token: string;
  post: Post;
  readonly clock: () => number; // seconds
  readonly queueMax: number;
  readonly maxAgeS: number;
  onEvent: (e: SendEvent) => void;

  constructor(apiUrl: string, deviceId: string, token: string, post: Post, clock: () => number, queueMax = 500,
    maxAgeS = 120, onEvent: (e: SendEvent) => void = () => {}) {
    this.url = `${apiUrl.replace(/\/+$/, "")}/api/ingest/train/${deviceId}`;
    this.deviceId = deviceId;
    this.token = token;
    this.post = post;
    this.clock = clock;
    this.queueMax = queueMax;
    this.maxAgeS = maxAgeS;
    this.onEvent = onEvent;
  }

  /** Point the device at another backend (or token) without losing its queue or sequence. */
  setTarget(apiUrl: string, token: string) {
    this.url = `${apiUrl.replace(/\/+$/, "")}/api/ingest/train/${this.deviceId}`;
    this.token = token;
  }

  /** A new reading with the next sequence number and the sender clock's time. */
  reading(fields: Omit<CarCount, "v" | "seq" | "ts">): CarCount {
    this.seq += 1;
    const msg: CarCount = { v: 1, seq: this.seq, ts: new Date(this.clock() * 1000).toISOString(), ...fields };
    for (const k of Object.keys(msg) as (keyof CarCount)[]) if (msg[k] === undefined || msg[k] === null) delete msg[k];
    return msg;
  }

  /** Queue a reading and try to send everything waiting. True when the queue is empty afterwards. */
  send(msg: CarCount): Promise<boolean> {
    this.queue.push({ at: this.clock(), msg });
    while (this.queue.length > this.queueMax) {
      const old = this.queue.shift()!;
      this.dropped += 1;
      this.onEvent({ at: this.clock(), seq: old.msg.seq, kind: "dropped", detail: "Queue full" });
    }
    return this.flush();
  }

  /** Send what is waiting, in order. Only one flush runs at a time. */
  flush(): Promise<boolean> {
    if (!this.flushing) this.flushing = this.flushOnce().finally(() => (this.flushing = null));
    return this.flushing;
  }

  private async flushOnce(): Promise<boolean> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) headers["X-Ingest-Token"] = this.token;
    while (this.queue.length) {
      const { at, msg } = this.queue[0];
      if (this.clock() - at > this.maxAgeS) {
        this.queue.shift();
        this.dropped += 1;
        this.onEvent({ at: this.clock(), seq: msg.seq, kind: "dropped", detail: "Older than 2 minutes" });
        continue;
      }
      let status: number;
      let body: unknown;
      try {
        ({ status, body } = await this.post(this.url, JSON.stringify(msg), headers));
      } catch {
        this.onEvent({ at: this.clock(), seq: msg.seq, kind: "offline" });
        return false;
      }
      if (status >= 500 || status === 429) {
        this.onEvent({ at: this.clock(), seq: msg.seq, kind: "retry", status });
        return false;
      }
      this.queue.shift();
      if (status >= 400) {
        this.rejected += 1;
        this.onEvent({ at: this.clock(), seq: msg.seq, kind: "rejected", status });
      } else {
        this.sent += 1;
        const accepted = (body as { accepted?: boolean } | undefined)?.accepted;
        this.onEvent({ at: this.clock(), seq: msg.seq, kind: "sent", status,
          detail: accepted === false ? "Received, but AERA could not match the train" : undefined });
      }
    }
    return true;
  }
}
