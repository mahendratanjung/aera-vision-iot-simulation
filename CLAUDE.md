# AERA Vision: project context

Simulation of AERA's in-train camera sensors (source 3 of AERA's observed crowding), backed by a
people-counting model whose accuracy is measured on held-out data. Approach and decisions: `PLAN.md`
(the "Decisions, 13 Nov 2026" block at the top overrides the draft below it). Run instructions: `README.md`.

## Ground rules

- **Mahendra is the sole author.** Never add Claude/Anthropic attribution anywhere: no
  `Co-Authored-By`, no "Generated with Claude Code", no author fields. Do not commit or push unless asked.
- **Never edit the AERA repo** (`../aera-advanced ecosystem-of-rail-analytics`). Read it only.
- **Never make up a metric.** Every number comes from a script that was run, on data the model never
  trained on, and is read from `model/results/`. The report, the README results paragraph and the
  app's Model page are generated from it. Say what could not be measured.
- **Never run heavy compute on the user's Mac** (model inference over many images, benchmarks, several
  servers or browsers at once) without asking first and saying how long and how heavy. On 13 Nov 2026
  the full inference run (plus a second parallel worker, a dev server and a headless browser) overloaded
  the laptop and it restarted. Prefer published figures, saved outputs, or a free Colab GPU.
- Ask before downloading anything over ~2 GB, any training run over ~1 hour, or signing up for any
  service. Everything stays free. Disk is tight (about 36 GB free on 13 Nov 2026).
- English crowding labels only (Quiet, Comfortable, Busy, Crowded, Packed). No Malay, no emoji.
- Design: AERA tokens only (`app/styles/theme.css`, copied from AERA), Plus Jakarta Sans, Phosphor
  icons, white background, 16 px cards, pills. Product voice: capital after "·" or ":".
- Chart series colours in fixed order: `--accent`, `--highlight`, `--accent-2` (validated); a reference
  line is dashed `--ink-3`. Every chart has a legend for 2+ series and a table or text equivalent.
- Downloaded checkpoints are loaded only with safe loaders (`torch.load(weights_only=True)`, ONNX).

## Layout

- `model/` (Python 3.12, uv): `aera_vision/` candidates, `infer` (cache raw outputs), `bench` (speed),
  `evaluate` (tune-split choices, winner rule, test scores, bootstrap, calibration), `report`
  (MODEL_REPORT.md, charts, README paragraph), `demo` (app photo bank), `serve` (local live counter).
  `vendor/zip_models` is ZIP's MIT model code. `data/splits.json` is the frozen split (hash-checked).
- `app/` (Next.js 16, TS, Tailwind 4, zustand): `lib/device` (smoother, sender, contract, confidence:
  ports of AERA's edge code), `lib/sim` (line model from AERA's KJ data, engine), screens Train,
  Camera, Model, Settings; `app/api/forward` forwards to AERA (only AERA's sensor paths).
  `scripts/sync-results.mjs` copies model/results into `generated/` before dev/build
  (`AERA_VISION_RESULTS=<dir>` overrides it for development only).
- `make data | eval | demo | app | live | check` (see Makefile).

## Key facts from AERA (read 13 Nov 2026)

- Message: `CarCount` (`packages/contracts/car-count.schema.json`), `POST /api/ingest/train/{device}`,
  header `X-Ingest-Token`. Readings with no matching train are dropped as `no_train`, so Send to AERA
  follows the trip AERA lists in `/api/stations/{id}/next-trains`. AERA's CORS allows only its own
  origin, hence the forwarder.
- AERA turns `confidence` into error: `sd = 8 / max(0.3, confidence)` points of car capacity (200 per
  car). The device sends confidence calibrated from the measured RMSE per level (tune split).
- Levels: Quiet <20 %, Comfortable 20–50, Busy 50–75, Crowded 75–95, Packed 95 %+.
- Sender: queue 500, drop readings older than 120 s, retry on 5xx/429/offline. Smoother: median of 9,
  send on a change of 3 or more, or every 30 s.

## Current state

- 13 Nov 2026: Part 1 and Part 2 built. No training (user's decision): 14 ready-trained candidates plus
  a hybrid, measured on CrowdHuman val (tune 1,000 complete; test 3,120 of 3,370 run, chunk 10 never
  ran because the laptop restarted; scored with `evaluate --available-only`, stated in the report).
  Chosen: hybrid of YOLOv5m CrowdHuman heads (below 41 people per view) and ZIP Nano NWPU (above).
  Speed not measured (bench not run, by decision after the crash). Demo bank: 79 test photos, every one
  checked at full size for watermarks (`data/demo_reviewed.txt` allow-list, `data/demo_exclude.txt`).
- 13 Nov 2026 (later): user found the app too complex. Now one page (`app/app/page.tsx`): camera view
  (blurred crowd photos, or own webcam unblurred) plus an About the model card. Train simulation, Model
  charts, Settings and the AERA forwarder moved to `app/app/_extras/` (private folder, not routed).
- 13 Nov 2026 (later still): page = live camera, photo upload count (`components/UploadCount.tsx`,
  local counter only), mixed-level example photos, corporate-style model performance (KPI strip, model
  table, 4 figures), hardware table. Hybrid = head detector below 41 people, ZIP Nano at 41+ (verified
  in `chosen.py`/`serve.py`; both run every frame, one count is used).
- 13 Nov 2026 (latest): one model only, YOLOv5m CrowdHuman heads (`model/chosen.json`, read by
  `chosen.winner()` for demo and live counter); page cut to live camera, photo upload, example photos;
  model performance and hardware moved to the user's report; report figures in `figures/`
  (`aera_vision/figures.py`, `make figures`).
- Open: the UI has not been viewed in a browser since the real results went in (typecheck passes);
  the last 250 test photos and the speed test would need a Colab GPU; nothing committed yet.

## Session log

- **2026-11-13** Read AERA and the live app; wrote PLAN.md. User decisions: no training, use
  ready-trained models and measure them ourselves; test set = CrowdHuman val (2.5 GB, approved);
  one camera sees half a car (100 people); demo footage picked by me from open sources. Downloaded
  CrowdHuman val (SHA-256 matches the authors' Hugging Face copy) and weights for YOLO11/26 n/s/m,
  RF-DETR N/S/M, YOLOv5m CrowdHuman (ONNX), ZIP N/T. Built model/ (metrics tested by hand and against
  pycocotools; AERA parity tests) and app/ (device logic ported from AERA with its test cases).
  Checked the UI in headless Chrome at 1440 and 390 px (no sideways scroll, no console errors).
  Found that some CrowdHuman photos carry stock-photo watermarks: excluded from the demo bank by
  `model/data/demo_exclude.txt`. Full inference (with a second parallel worker added by me) overloaded
  the laptop and it restarted; user asked for no more heavy work locally. Scored the saved outputs
  (3,120 test photos), wrote the report, rebuilt the demo bank. Send to AERA tested end to end
  against a scratch copy of AERA's backend (readings accepted and fused as 4 camera observations).
