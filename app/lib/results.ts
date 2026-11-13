// Shapes of model/results/summary.json as copied by scripts/sync-results.mjs. Every number shown on
// the Model page comes from that file; nothing is typed here.
import raw from "@/generated/results.json";

export type CI = [number, number];
export interface Counting { n: number; mae: number; rmse: number; mape: number | null; mean_signed_error: number; n_zero_true: number }
export interface PerLevel { level: number; name: string; support: number; predicted: number; precision: number; recall: number; f1: number }
export interface ByLevel { level: number; name: string; n: number; mae: number | null; rmse: number | null; mape: number | null; mean_signed_error: number | null; level_recall: number | null }
export interface Split {
  counting: Counting;
  levels: { per_level: PerLevel[]; macro_f1: number; accuracy: number; within_one: number; confusion: number[][] };
  by_level: ByLevel[];
  ci95?: { mae: CI; rmse: CI; mean_signed_error: CI; level_macro_f1: CI; level_accuracy: CI };
  crowded_packed_mean_signed_error?: number | null;
  per_image?: { true: number[]; pred: number[] };
}
export interface Detection {
  threshold: number; precision: number; recall: number; f1: number; ap50: number; ap50_95: number; ap50_full_body?: number;
  pr_curve: { recall: number[]; precision: number[]; score: number[] };
  ci95?: { precision: CI; recall: CI; f1: CI };
}
export interface Candidate {
  key: string; label: string; kind: "person" | "head" | "density" | "hybrid"; family: string; licence: string; trained_on: string;
  source: string; note: string; params_m: number | null; file_mb: number | null; cpu_fps: number | null; mps_fps: number | null;
  threshold: number | null; parts?: string[]; switch?: number;
  test_detection?: Detection; tune: Split; test: Split;
}
export interface Band { level: number; name: string; n: number; rmse_people: number | null; bias_people: number | null; rmse_used: number | null;
  borrowed?: boolean; car_sd_points_if_both_halves: number | null; confidence_if_both_halves: number }
export interface SpeedRow { file_mb: number; params_m: number | null; cpu?: { ms_median: number; fps: number }; mps?: { ms_median: number; fps: number } }
export interface Results {
  missing?: boolean;
  generated: string; command: string;
  dataset: { name: string; url: string; homepage: string; licence: string; images: number; tune: number; test: number; split_seed: number;
    splits_sha256: string; test_planned?: number; test_not_run?: number; test_note?: string | null; test_levels: Record<string, number>; tune_levels: Record<string, number>; people_counted_test: number };
  rules: { view_capacity: number; car_capacity: number; iou: number; threshold: string; score_floor: number;
    bootstrap: { resamples: number; seed: number }; speed_rule_cpu_fps: number; speed_rule_applied?: boolean; speed_note?: string | null; tie_macro_f1: number; macro_f1: string };
  winner: { key: string; label: string; excluded_too_slow: string[]; tied_on_macro_f1: string[]; best_tune_macro_f1: number };
  calibration: { bands: Band[]; rule: string };
  candidates: Candidate[];
  speed: { machine: { chip: string; platform: string; torch: string; threads: number }; frame: { size: number[] }; candidates: Record<string, SpeedRow> } | null;
  edge: { note: string; devices: { device: string; figure: string; source: string; url: string }[]; needed: string } | null;
}

export const results = raw as unknown as Results;
