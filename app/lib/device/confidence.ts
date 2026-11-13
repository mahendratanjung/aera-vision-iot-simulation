/**
 * The `confidence` a car's device sends, set from the measured counting error.
 *
 * AERA reads confidence as an error size: sd = 8 / max(0.3, confidence) points of the car's
 * capacity (crowdsource/sensors.py). Our device adds two camera views; each view's error is the
 * RMSE measured on the tune split for the level that view's count falls in
 * (model/results/summary.json, calibration.bands). Treating the two errors as independent, the
 * car's error is sqrt(e_a^2 + e_b^2) people, which is that / 200 * 100 points.
 */
import { CAR_CAPACITY, VIEW_CAPACITY } from "../sim/constants.ts";
import { crowdOf } from "../crowd/levels.ts";

export const AERA_CAMERA_SD = 8;
export const AERA_MIN_CONFIDENCE = 0.3;

export interface Band { level: number; rmse_used: number }

export function viewError(count: number, bands: Band[]): number {
  const level = crowdOf((count / VIEW_CAPACITY) * 100).level;
  const b = bands.find((x) => x.level === level);
  return b && Number.isFinite(b.rmse_used) ? b.rmse_used : Math.max(...bands.map((x) => x.rmse_used).filter(Number.isFinite));
}

/** Car error in points of capacity, and the confidence that makes AERA use exactly that error. */
export function carConfidence(countA: number, countB: number, bands: Band[]): { sdPoints: number; confidence: number } {
  const ea = viewError(countA, bands);
  const eb = viewError(countB, bands);
  const sdPoints = (Math.sqrt(ea * ea + eb * eb) / CAR_CAPACITY) * 100;
  const raw = sdPoints > 0 ? AERA_CAMERA_SD / sdPoints : 1;
  return { sdPoints, confidence: Math.round(Math.min(1, Math.max(AERA_MIN_CONFIDENCE, raw)) * 100) / 100 };
}
