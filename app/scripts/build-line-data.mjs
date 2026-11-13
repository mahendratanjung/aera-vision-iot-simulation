// Build data/kj.json from AERA's line and crowding model (read only):
//   ../aera-advanced ecosystem-of-rail-analytics/data/lines/KJ.json   stations, run times, headways
//   ../aera-advanced ecosystem-of-rail-analytics/data/models/KJ.json  passengers per hour on each segment
// Run: node scripts/build-line-data.mjs   (the output is committed, so the app runs without AERA)
import { readFileSync, writeFileSync } from "node:fs";

const AERA = new URL("../../../aera-advanced ecosystem-of-rail-analytics/", import.meta.url);
const line = JSON.parse(readFileSync(new URL("data/lines/KJ.json", AERA), "utf8"));
const model = JSON.parse(readFileSync(new URL("data/models/KJ.json", AERA), "utf8"));

const out = {
  source: {
    note: "Built from AERA's data/lines/KJ.json (Prasarana GTFS Static) and data/models/KJ.json (data.gov.my ridership).",
    model_window: model.window,
    model_source: model.source,
    built: new Date().toISOString().slice(0, 10),
  },
  capacity: model.capacity,
  stations: line.stations.map((s) => ({ id: s.id, name: s.name })),
  directions: line.directions.map((d) => ({
    direction_id: d.direction_id,
    headsign: d.headsign,
    stops: d.stops.map((s) => ({ station_id: s.station_id, arr: s.arr, dep: s.dep })),
  })),
  frequencies: line.frequencies,
  calendar: line.calendar,
  // load[weekday][direction][segment][hour]: passengers per hour; weekday 0 = Monday;
  // segment s joins stations[s] and stations[s + 1]
  load: model.load,
  board: model.board,
};
writeFileSync(new URL("../data/kj.json", import.meta.url), JSON.stringify(out));
console.log(`data/kj.json: ${out.stations.length} stations, ${Object.keys(out.load).length} weekdays`);
