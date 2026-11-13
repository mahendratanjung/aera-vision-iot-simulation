/**
 * Turns noisy per-frame people counts into steady readings worth sending. A line-for-line port of
 * AERA's edge/aera_edge/smoothing.py (CountSmoother): the median of the last `window` counts,
 * reported when it moves by `minChange` or more, or every `everyS` seconds.
 */
export class CountSmoother {
  private counts: number[] = [];
  lastSent: number | null = null;
  private lastSentAt = -1e18;

  readonly window: number;
  readonly minChange: number;
  readonly everyS: number;

  constructor(window = 9, minChange = 3, everyS = 30) {
    this.window = window;
    this.minChange = minChange;
    this.everyS = everyS;
  }

  /** Add one frame's count at time `nowS` (seconds). Returns a value to send, or null. */
  add(count: number, nowS: number): number | null {
    this.counts.push(count);
    if (this.counts.length > this.window) this.counts.shift();
    if (this.counts.length < this.window) return null;
    const value = roundHalfEven(median(this.counts));
    const changed = this.lastSent === null || Math.abs(value - this.lastSent) >= this.minChange;
    if (changed || nowS - this.lastSentAt >= this.everyS) {
      this.lastSent = value;
      this.lastSentAt = nowS;
      return value;
    }
    return null;
  }
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Python's round(): halves go to the even neighbour (round(2.5) == 2). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
