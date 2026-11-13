/** Our copy of levels.ts must stay identical to AERA's, and the thresholds must be the agreed ones. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { CROWD, crowdOf, DEFAULT_BOUNDS, pctText } from "./levels.ts";

test("thresholds and names", () => {
  assert.deepEqual(DEFAULT_BOUNDS, [20, 50, 75, 95]);
  assert.deepEqual(CROWD.map((c) => c.name), ["Quiet", "Comfortable", "Busy", "Crowded", "Packed"]);
  const cases: [number, string][] = [[0, "Quiet"], [19.99, "Quiet"], [20, "Comfortable"], [49.9, "Comfortable"], [50, "Busy"],
    [74.9, "Busy"], [75, "Crowded"], [94.9, "Crowded"], [95, "Packed"], [140, "Packed"]];
  for (const [pct, name] of cases) assert.equal(crowdOf(pct).name, name, `${pct}%`);
  assert.equal(pctText(100), "over capacity");
});

test("file matches AERA's levels.ts apart from the copy note", (t) => {
  const ours = readFileSync(new URL("./levels.ts", import.meta.url), "utf8").split("\n").slice(1).join("\n");
  let theirs: string;
  try {
    theirs = readFileSync(new URL("../../../../aera-advanced ecosystem-of-rail-analytics/frontend/lib/crowd/levels.ts", import.meta.url), "utf8");
  } catch {
    t.skip("AERA repo not found");
    return;
  }
  assert.equal(ours, theirs);
});
