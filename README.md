# AERA Vision

A simulation of AERA's third crowding source: a camera counter in every car of a Kelana Jaya Line
train. It counts the people in each car and sends AERA only the number, never an image. It comes
with the measured accuracy of the people-counting model it uses.

AERA (Advanced Ecosystem of Rail Analytics) is a digital twin of the LRT Kelana Jaya Line that shows
how crowded every train is ([live app](https://aera-rail.vercel.app)). It already takes rider reports
and opt-in phone locations. This project shows how in-train cameras would work, without buying hardware.

## Results in one paragraph (for the pitch deck)

<!-- results:start -->
We compared 14 ready-trained people counters (no retraining), plus a hybrid of the best two, on 3,120 held-out photos of crowds from CrowdHuman validation set (the other 250 planned test photos were not run). The chosen model is **Hybrid: YOLOv5m CrowdHuman, heads below 41, ZIP Nano (NWPU) above**. On the test photos it puts a camera view in the right crowding level 84.3% of the time (82.9% to 85.5%) (macro-F1 0.65 (0.61 to 0.69)) and miscounts by 5.0 people on average (4.7 to 5.4). Its head detector (YOLOv5m CrowdHuman, heads) finds people with precision 91.0% (90.7% to 91.4%), recall 71.1% (69.6% to 72.5%) and F1 79.8% (78.8% to 80.8%) at IoU 0.5. The person detector AERA's edge code uses today (YOLO11n) scores macro-F1 0.31 (0.30 to 0.32) and miscounts by 8.0 people (7.4 to 8.6). The chosen model's hardest level is Packed (33.8 people off on average). Only 97 test photos are Crowded or Packed, so those numbers are the least certain.
<!-- results:end -->

Full method, tables and charts: [model/MODEL_REPORT.md](model/MODEL_REPORT.md). What the model is and
isn't for: [model/MODEL_CARD.md](model/MODEL_CARD.md).

## How it connects to AERA

| | AERA Vision | AERA |
| --- | --- | --- |
| Message | Builds `CarCount` readings: `v`, `seq`, `ts`, `method: "camera"`, `car`, `count`, `confidence`, and `trip_id` or `station_id` + `direction_id` | `packages/contracts/car-count.schema.json`, `backend/aera_twin/contracts.py` |
| Transport | Posts to `POST /api/ingest/train/{device}` with header `X-Ingest-Token` (through the app's own forwarder, since AERA only accepts browser calls from its own site) | `backend/aera_twin/api/routes.py`; MQTT `aera/v1/train/{device}/count` takes the same JSON |
| Device rules | Median of the last 9 frames, sent on a change of 3 or more people or every 30 s; queue of 500; readings older than 2 minutes dropped | Same rules as `edge/aera_edge/smoothing.py` and `sender.py` |
| Trust | `confidence` is set so AERA's rule `sd = 8 / max(0.3, confidence)` gives exactly the counting error we measured | `backend/aera_twin/crowdsource/sensors.py`, `fusion.py` |
| Levels | Quiet, Comfortable, Busy, Crowded, Packed at 20 / 50 / 75 / 95 % | `frontend/lib/crowd/levels.ts` (copied, and a test checks it still matches) |

The simulation works without AERA. **Send to AERA** (Settings, off by default) posts every reading to a
local (`http://localhost:8000`) or deployed AERA backend. It follows the next train AERA expects at the
chosen station, so AERA can match the readings to a real trip.

## Run it

Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 24 and about 6 GB of free disk.

```bash
make data    # CrowdHuman validation set + published model weights, checksums verified (about 3.4 GB)
make score   # score the saved model outputs, write model/results/ and model/MODEL_REPORT.md (light)
# make eval  # heavy: rerun every model from scratch (see below)
make demo    # build the app's camera photos (test split, faces blurred) with the chosen model's output
cd app && npm install && cd ..
make app     # http://localhost:3100
make live    # optional: runs the model on your webcam twice a second (moderate load; stop it with Ctrl+C when done)
make check   # ruff + pytest, eslint + node tests + tsc
```

**`make eval` is heavy:** about 90 minutes of full CPU and GPU on an M4 Pro MacBook (14 models over
4,370 photos). On 13 Nov 2026 it overloaded the laptop, which restarted before the last 250 test photos
were run. The published results score the 3,120 test photos that were run (the report says so). To
rescore the saved outputs without running any model: `make score` (light, a few minutes on one core).
To run the rest, use a free GPU such as Google Colab rather than the laptop. The speed test
(`aera_vision.bench`) was not run, for the same reason.

### The app

One page, three sections:

1. **Live camera:** your own webcam, with a box on each head and the count.
2. **Count people in a photo:** choose or drop any photo; it is counted on this computer and not saved.
3. **Example photos:** 12 held-out test photos from Quiet to Packed, faces blurred (real people from the
   internet), with the model's count and the actual count.

Start everything with one command, from the project folder (or from `app/`):

```bash
npm run dev    # website on http://localhost:3100 + the counter for the camera and photo upload
```

Wait for "Counter ready" in the terminal, then open http://localhost:3100. Ctrl+C stops both.

**Model:** one model, YOLOv5m trained on CrowdHuman, counting heads (`model/chosen.json` says why).
**Figures for the report:** `figures/` (precision-recall, F1/precision/recall by confidence, confusion
matrix, counted vs actual, comparison of all detectors, error by level), each with a caption in
`figures/README.md`. Rebuild with `make figures`.

The earlier, larger version (train simulation with scenarios, charts page, Send to AERA) is kept in
`app/app/_extras/`, which the app does not show.

## Reproduce the metrics

Everything is in `model/`. `data/splits.json` freezes the split into 1,000 tune and 3,370 test photos,
and its hash is checked before every run. Then:

1. `aera_vision.infer` runs every model on every photo and caches raw outputs.
2. `aera_vision.bench` measures speed.
3. `aera_vision.evaluate` sets thresholds on tune only, picks the model by the rule fixed in
   `PLAN.md`, scores the test split once with bootstrap intervals, and writes `results/`.
4. `aera_vision.report` writes the report, the charts and the paragraph above.

The metric functions are tested on small examples worked out by hand, and average precision is
cross-checked against pycocotools.

## Repository

```
model/   Python: candidates/, metrics, evaluation, report, demo bank, live counter, tests
app/     Next.js + TypeScript: device logic (smoother, sender, contract), simulation, screens, tests
contracts/car-count.schema.json   copy of AERA's contract (a test checks it matches)
PLAN.md  approach and decisions;  CLAUDE.md  context and session log
```

## Credits and licences

- Test data: CrowdHuman (Shao et al., 2018), CC BY-NC 4.0, non-commercial research. Demo photos are
  from its validation set, faces blurred.
- Models: Ultralytics YOLO11 / YOLO26 (AGPL-3.0); RF-DETR (Apache-2.0); YOLOv5m CrowdHuman ONNX
  export by yakhyo (no licence file; YOLOv5 is AGPL-3.0); ZIP by Yiming Ma (MIT, model code vendored
  in `model/aera_vision/vendor/zip_models` with its licence).
- Line data and crowding model: from AERA (Prasarana GTFS Static, data.gov.my ridership).
- Brand: AERA Express logo from AERA's `brand/`.
