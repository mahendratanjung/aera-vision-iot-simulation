// Copy the evaluation's output (model/results/, written by `make eval`) into the app.
// Nothing here computes or types a number: it only selects fields so pages load less.
// Runs before `npm run dev` and `npm run build`. If the evaluation has not been run, the app says so.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

// AERA_VISION_RESULTS may point elsewhere for development only; real builds read model/results.
const RESULTS = process.env.AERA_VISION_RESULTS ? new URL(`file://${process.env.AERA_VISION_RESULTS.replace(/\/?$/, "/")}`) : new URL("../../model/results/", import.meta.url);
const REFS = new URL("../../model/references/", import.meta.url);
const OUT = new URL("../generated/", import.meta.url);
mkdirSync(OUT, { recursive: true });

const read = (u) => (existsSync(u) ? JSON.parse(readFileSync(u, "utf8")) : null);
const summary = read(new URL("summary.json", RESULTS));
const speed = read(new URL("speed.json", RESULTS));
const edge = read(new URL("edge_speed.json", REFS));

if (!summary) {
  writeFileSync(new URL("results.json", OUT), JSON.stringify({ missing: true }));
  writeFileSync(new URL("calibration.json", OUT), JSON.stringify({ missing: true }));
  console.log("sync-results: model/results/summary.json not found. Run `make eval` first; the Model page will say so.");
  process.exit(0);
}

const winnerKey = summary.winner.key;
const slim = {
  ...summary,
  speed,
  edge,
  candidates: summary.candidates.map((c) => {
    const keepPoints = c.key === winnerKey || c.key === "yolo11n";
    const strip = (part) => (part ? { ...part, per_image: keepPoints ? part.per_image : undefined } : part);
    return { ...c, tune: { ...strip(c.tune), per_image: undefined }, test: strip(c.test) };
  }),
};
writeFileSync(new URL("results.json", OUT), JSON.stringify(slim));
const w = summary.candidates.find((c) => c.key === winnerKey);
writeFileSync(new URL("calibration.json", OUT), JSON.stringify({
  generated: summary.generated,
  winner: { key: w.key, label: w.label, kind: w.kind, threshold: w.threshold ?? null },
  bands: summary.calibration.bands.map((b) => ({ level: b.level, name: b.name, n: b.n, rmse_used: b.rmse_used })),
}));
console.log(`sync-results: ${summary.candidates.length} candidates, winner ${w.label}, generated ${summary.generated}`);
