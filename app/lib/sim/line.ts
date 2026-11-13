/**
 * The Kelana Jaya Line for the simulation: stations, run times, headways and AERA's typical
 * crowding, from data/kj.json (built from AERA's data by scripts/build-line-data.mjs).
 *
 * A train's load on the leg leaving stop k follows AERA's frontend/lib/crowd/model.ts:
 *   load % = passengers per hour on that segment x scheduled headway / 3600 / capacity x 100
 * with the hour interpolated, plus an optional event surge boarding at one station that fades
 * by 0.82 per stop downstream (AERA's surge rule).
 */

export interface Stop { station_id: string; arr: number; dep: number }
export interface Direction { direction_id: 0 | 1; headsign: string; stops: Stop[] }
export interface LineData {
  capacity: number;
  stations: { id: string; name: string }[];
  directions: Direction[];
  frequencies: Record<string, Record<string, { start: number; end: number; headway: number }[]>>;
  calendar: Record<string, number[]>;
  load: Record<string, Record<string, number[][]>>;
  board: Record<string, Record<string, number[][]>>;
}

export interface Surge { stationId: string; multiplier: number }

export class Line {
  readonly data: LineData;
  readonly idx: Record<string, number>;

  constructor(data: LineData) {
    this.data = data;
    this.idx = Object.fromEntries(data.stations.map((s, i) => [s.id, i]));
  }

  name(stationId: string): string {
    return this.data.stations[this.idx[stationId]]?.name ?? stationId;
  }

  dir(d: 0 | 1): Direction {
    return this.data.directions.find((x) => x.direction_id === d)!;
  }

  /** Index of a station in a direction's stop list. */
  stopIndex(d: 0 | 1, stationId: string): number {
    return this.dir(d).stops.findIndex((s) => s.station_id === stationId);
  }

  /** Segment joining stop k and k+1 of a direction (AERA's segment()). */
  segment(d: 0 | 1, k: number): number {
    const stops = this.dir(d).stops;
    return Math.min(this.idx[stops[k].station_id], this.idx[stops[k + 1].station_id]);
  }

  /** Scheduled seconds between trains at time of day t on weekday wd, or null outside service. */
  headway(d: 0 | 1, wd: number, t: number): number | null {
    const service = Object.entries(this.data.calendar).find(([, days]) => days[wd] === 1)?.[0];
    const windows = service ? this.data.frequencies[service]?.[String(d)] ?? [] : [];
    for (const w of windows) if (t >= w.start && t < w.end) return w.headway;
    return null;
  }

  /** Passengers per hour, linear between hour centres (AERA's interp in lib/crowd/model.ts). */
  flow(table: "load" | "board", d: 0 | 1, wd: number, i: number, t: number): number {
    const row = this.data[table][String(wd)][String(d)][i];
    const x = t / 3600 - 0.5;
    const h0 = Math.floor(x);
    const f = x - h0;
    const a = row[((h0 % 24) + 24) % 24];
    const b = row[(((h0 + 1) % 24) + 24) % 24];
    return a + (b - a) * f;
  }

  /** Typical train load in % of capacity leaving stop k of direction d at time t (seconds of day). */
  legLoad(d: 0 | 1, k: number, wd: number, t: number, surge?: Surge): number {
    const stops = this.dir(d).stops;
    if (k >= stops.length - 1) return 0;
    const h = this.headway(d, wd, t) ?? 420;
    let load = (this.flow("load", d, wd, this.segment(d, k), t) * h) / 3600 / this.data.capacity * 100;
    if (surge && surge.multiplier > 1) {
      const x = this.stopIndex(d, surge.stationId);
      if (x >= 0 && x <= k) {
        const extra = (surge.multiplier - 1) * this.flow("board", d, wd, this.idx[surge.stationId], t) * h / 3600;
        load += (extra * Math.pow(0.82, k - x) / this.data.capacity) * 100;
      }
    }
    return load;
  }
}
