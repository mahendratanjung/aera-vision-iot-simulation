# AERA Vision: plan

## Decisions, 13 Nov 2026 (these override the draft below where they differ)

- **No training.** Only ready-trained models are used. We still measure every one ourselves, by
  running it (inference only) on a labelled test set it was not trained on. The authors' published
  numbers are shown alongside and labelled as published.
- **Test set: CrowdHuman validation** (4,370 crowded photos, every person boxed, plus head boxes;
  2.5 GB; non-commercial research licence). No hand labelling. It is split once, with a fixed seed,
  into **tune** (1,000 images: thresholds, hybrid switch point, confidence calibration) and **test**
  (the remaining 3,370: reported numbers only).
- **Level rule:** one camera sees half a car, so a view holds 100 people at full load. Level =
  count / 100 with AERA's thresholds (Quiet under 20, … Packed 95+), applied to truth and prediction
  alike.
- **Candidates** (all pretrained):
  - YOLO11 n/s/m and YOLO26 n/s/m (COCO, person class).
  - RF-DETR Nano/Small/Medium (COCO, person class).
  - YOLOv5m CrowdHuman person+head (community weights, `yakhyo/yolov5-crowdhuman-onnx`). Training
    on CrowdHuman train probably used the val split to pick checkpoints, so this is flagged as a
    possible slight advantage.
  - ZIP density counter (trained on ShanghaiTech/NWPU).
  - Hybrid of a detector and a counter.

  Self-trained head detectors (H1/H2) are dropped.
- **Demo footage:** open photos of congested people, picked by me: about 100 photos from the
  CrowdHuman **test** split, spread over counts 0 to 150, faces blurred before saving
  (`make demo`). Each carries its labelled count and the chosen model's real output, so every
  number in the app is a measured one. Photos stay out of git (CC BY-NC 4.0); a webcam panel
  gives a live demo through a local counter.
- **Speed rule** for picking the model: at least 2 fps on this Mac's CPU alone, as a stand-in for
  the edge device, which can't be measured here.
- Sections 2 (view capacity), 5 (our own train-interior labelling) and 6 (training) below are
  superseded by these decisions and kept only as history.

AERA Vision is a simulation of AERA's third crowding source: a camera in each car of a Kelana
Jaya Line train that counts the people inside and sends only the number to AERA as a `CarCount`.
It has two parts:

1. **`model/`**: picks the people-counting model by measuring every candidate the same way on a
   held-out set of train-interior images, and writes every number to `model/results/`.
2. **`app/`**: a web app that plays the sensors on one Innovia train (4 cars) using those images
   and the chosen model's real outputs. It shows exactly what leaves the car, and can optionally
   post the readings to an AERA backend.

The repo is this folder (`aera-iot-simulation`, already an empty git repo), not a new
`aera-vision` folder. The product name is **AERA Vision**. AERA itself is only read, never edited.

---

## 1. What AERA already decides for us (read from the AERA repo)

| Topic | What AERA does | What it means here |
| --- | --- | --- |
| Message | `CarCount` (`backend/aera_twin/contracts.py`, `packages/contracts/car-count.schema.json`): `v: 1`, `seq ≥ 0`, `ts`, `method`, `car` 1..4, exactly one of `count` (0..600) or `load_pct` (0..250), optional `confidence` 0..1, then `trip_id` or (`station_id` + `direction_id`). `additionalProperties: false`. | The sim sends `method: "camera"` with `count`. A test validates every message against a copy of the schema and checks the copy still matches AERA's file. |
| Endpoint | `POST /api/ingest/train/{device_id}`, header `X-Ingest-Token`, device id `^[A-Za-z0-9_-]+$` (max 64), body max 4096 bytes, answers `202 {"accepted": bool}`. MQTT `aera/v1/train/{device}/count` takes the same JSON. | Device ids like `cam-t07-c2`, the same style as `edge/README.md`. |
| Matching | `find_train` uses the `trip_id`, or the single train standing at that station in that direction. If nothing matches, AERA counts the reading as `no_train` and drops it. | When "Send to AERA" is on, the sim attaches to a train that is actually running in that AERA instance (from its live state), or AERA will reject the readings. |
| Error model | `sensors.car_observation`: car load % = `count / 200 × 100`; error `sd = 8.0 / max(0.3, confidence)` percentage points (camera base sd is 8). Fused by inverse variance with a 4-minute half-life. | **`confidence` is an error statement, not a detector score.** Today `camera_counter.py` sends the mean box score, which is not calibrated. Here, `confidence = clamp(8 / σ, 0.3, 1)`, where σ is the **measured** counting RMSE in points of car capacity for the predicted level band, taken from the validation split. The 0.3 floor means AERA cannot express an error above about 27 points. If we measure more than that at Packed, the report says so as a finding for AERA. |
| Levels | `levels.ts`: Quiet < 20 %, Comfortable 20–50, Busy 50–75, Crowded 75–95, Packed 95 %+, over 100 % shown as "over capacity". Car capacity 200 (800 per train). | A test parses AERA's `levels.ts`, when the sibling repo is present, and checks our thresholds against it. |
| Smoothing | `CountSmoother`: median of the last 9 frame counts; send when it moves by 3 or more, or every 30 s. | Ported as is, with the same test cases. |
| Sender | `Sender`: `seq` +1 per reading; queue of up to 500; readings older than 120 s are dropped; 5xx/429 or network error keeps the queue; other 4xx counts as rejected and is not retried. | Ported as is, with an "offline" switch in the UI that shows the queue filling, draining and dropping. |
| Camera code | `camera_counter.py`: YOLO11n person class, region of interest, about 2 fps, preview blurred, frames never stored. | `yolo11n` is the baseline every candidate must beat. |
| Look | `theme.css`, `CrowdTag`/`CrowdMeter`/`CrowdIcon`/`TrainCars` in `components/Crowd.tsx`; the live app (glance card with the joined-car strip, white cards, pills, dark primary button). | Tokens copied into one `app/styles/theme.css`; the components are rebuilt with the same look. |

Context worth one line in the pitch: in September 2025 Rapid Rail said it would bring a crowd
monitoring system that shows which coaches are full before the train arrives, without saying how
it would measure them (Lowyat.net, 22 Sep 2025).

---

## 2. The measurement problem to settle first

**A camera sees part of a car. A photo sees part of a car.** An Innovia car is about 17 m long.
One camera at standing height or on the ceiling at one end cannot see all 200 people of a Packed
car; heads far down the aisle are a few pixels wide and hidden behind closer people. Public photos
of train interiors usually show a third to a half of a car. So "level = count / 200" only holds
when the cameras cover the whole car. Applied to a photo of half a car, it labels a visibly Packed
scene as Comfortable.

Proposal:

- **Hardware assumption (written into the model card):** two cameras per car, one at each end,
  looking along the aisle, each counting only its half (region of interest split at the middle,
  so no one is counted twice). The car's device adds the two halves and sends one `CarCount`. In
  the app each car still shows as one sensor, with "2 cameras" in its status.
- **Each test scene gets a view capacity `C`:** how many people the visible part would hold at
  100 %, that is `200 × the share of the car in view`. The share is labelled once per scene (for
  a video, once per video, since the camera is fixed) on a five-step scale: 1/4, 1/3, 1/2, 2/3,
  whole car.
  - True level = level(true count / C). Predicted level = level(predicted count / C).
  - Because truth and prediction share `C`, the confusion matrix measures exactly what the sensor
    adds: how often counting error moves a car across a level boundary.
  - For footage covering a whole car, `C = 200`, so this is your count / 200 rule.
- **An eye-judged level for every frame**, using AERA's own descriptions ("Plenty of seats" …
  "Boarding may not be possible"). It is a sanity check on `C`: we report how often it agrees
  with level(true count / C).

---

## 3. Candidate models

All candidates run through one interface: `count(image) → number + optional boxes or points`.
Parameter counts and file sizes in the report come from the eval script, not from this table.

| # | Candidate | Kind | Code licence | Weights trained on | Edge path | Why it is here |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | **YOLO11 n/s/m**, person class | Body detector, zero-shot | AGPL-3.0 (Ultralytics) | COCO | Hailo model zoo / TensorRT | What AERA's edge code uses today: the baseline |
| D2 | **YOLO26 n/s/m**, person class | Body detector, zero-shot | AGPL-3.0 | COCO | Built for edge export (released Jan 2026) | Ultralytics' current default |
| D3 | **RF-DETR Nano/Small/Medium**, person class | Transformer detector, zero-shot | Apache-2.0 (sizes up to Large; XL/2XL use a separate platform licence) | COCO | ONNX / TensorRT; Hailo support not confirmed | Strongest permissively licensed real-time detector |
| H1 | **YOLO26n-head, YOLO26s-head** | **Head** detector, fine-tuned by us | AGPL-3.0; **dataset non-commercial** | CrowdHuman head boxes | As D2 | Heads stay visible when bodies hide each other |
| H2 | **RF-DETR Nano head** (stretch goal) | Head detector, fine-tuned by us | Apache-2.0; dataset non-commercial | CrowdHuman head boxes | As D3 | Permissive alternative to H1, only if training it on MPS/Colab works |
| C1 | **ZIP Nano / Tiny** (Ma et al., 2025) | Density counter, no boxes | MIT | ShanghaiTech B (also NWPU) | Small CNN (3.4 M / 10.5 M parameters per authors) | Newest counter with edge-sized variants |
| C2 | **DM-Count** (2020) | Density counter | MIT | ShanghaiTech B / NWPU | VGG-19, heavy | Classic, widely cited baseline |
| X1 | **Hybrid**: best head detector up to a switch point, best counter above it | Hybrid | Inherits both | n/a | Both models on the device | Detector precision when sparse, counter robustness when packed. The switch point is chosen on the validation split. |

Considered and left out:

- **P2PNet:** its licence limits use to academic research.
- **APGCC:** MIT, but only ShanghaiTech A weights (very dense outdoor crowds).
- **PET:** no licence file.
- **CSRNet:** no licence file, superseded.
- **DEIMv2:** custom licence.
- **RT-DETRv2 / D-FINE:** Apache-2.0, but RF-DETR is newer and stronger in the same family; added
  only if time allows.
- **Ready-made "CrowdHuman head" YOLO weights from GitHub/Hugging Face:** unclear training and
  licence. We train our own so the provenance is clean.

Licence note for the pitch: everything Ultralytics-based is AGPL-3.0. Shipping it on Prasarana's
trains would need an Ultralytics Enterprise licence, or the Apache/MIT path (RF-DETR + ZIP). The
report compares both paths, so the choice is visible.

---

## 4. Datasets

| Dataset | Use | Licence / terms | Size | Download? |
| --- | --- | --- | --- | --- |
| **CrowdHuman** (Megvii): 15,000 train / 4,370 val images, about 23 people per image; full, visible and head boxes | Train head detectors; head-box detection metrics on its val split | **Non-commercial research and education only** (per its terms) | Several GB; mirrors disagree on zip sizes. I'll check the exact official sizes before asking. | **Needs your OK** (over 2 GB). To save disk: train01 (about 5,000 images) + val only, not all three train zips |
| **ShanghaiTech Part B**: 716 street images (400 train / 316 test), head points | Sanity check: our pipeline should reproduce ZIP/DM-Count's published MAE on its test split | No stated licence; research use | Expected well under 2 GB | Yes, if under 2 GB |
| JHU-Crowd++ | Not needed: the pretrained counters already cover it, and it is about 2.9 GB | Research only | 2.87 GB | No |
| **Train-interior set (ours)**: see section 5 | **The test set that matters** | Per image (CC0 / CC BY / CC BY-SA / Pexels licence) or our own footage with consent | A few hundred MB at 1600 px | Yes, images under 2 GB |

Disk: this Mac has **40 GB free (91 % used)**. Data goes in `model/data/` (git-ignored). I'll keep
the total under about 15 GB and delete unzipped archives after conversion.

---

## 5. Train-interior test set (the credibility piece)

**Sources, in order of preference:**

1. **Our own footage (best):** a phone on a pole or mount at ceiling height at a car end, on Kelana
   Jaya trains. Only face-blurred frames are kept. This needs permission from Prasarana, a letter
   from UTP, and a consent notice. It is the only source with the deployment camera angle and
   whole-car coverage. **Your decision (section 14).**
2. **Wikimedia Commons:** free-text search returns about 13,700 files for "crowded train
   interior", 1,900 for "metro interior passengers" and 102 for "Rapid KL LRT interior" (searched
   today; noisy).
   - A script filters by licence (CC0, PD, CC BY, CC BY-SA only), size (≥ 800 px) and file type,
     then saves the source URL, author and licence for each image in `model/data/manifest.csv`.
   - I go through thumbnails and keep only real car interiors with passengers. This is curation,
     not labelling.
   - CC BY / BY-SA need attribution: a generated `CREDITS.md`.
3. **Pexels / Pixabay stock video** of crowded metro cars: free licence (not CC), fixed
   viewpoints, useful for Crowded and Packed. One frame every 5 s at most, to avoid near-duplicates.

**Expected weakness:** public material is mostly hand-held at eye level, and Packed scenes are
rare. If Packed ends up with fewer than about 40 frames, the report says so and the confidence
intervals show it. Without our own footage, the set cannot show the ceiling-camera angle.

**Labels (done by a person, not by me, and not pre-filled by any candidate model, which would
bias the result toward that model):**

- A point on every visible head, which gives the count.
- Per scene: the share of the car in view, which gives `C` (section 2).
- Per frame: the eye-judged level.
- Head boxes on about 100 frames (Quiet to Busy), for box metrics on train interiors.
- Tags: night / reflections / people seen through the doors / children / bags or prams.

**Tool:** Label Studio, run locally (open source, free, no sign-up, nothing leaves the Mac). I'll
supply the labelling setup and a one-page guide. **Rough effort:** at 1–2 s per head point, 400
frames averaging about 40 heads is about 16,000 clicks, or **6–9 hours** of labelling.

**Splits:** by source (photographer, video or recording session), never by frame, so near-identical
frames cannot sit on both sides.

| Split | Size | Used for |
| --- | --- | --- |
| Validation | About 100 frames | Thresholds, hybrid switch point, confidence calibration |
| Test | About 300 frames, spread over the five levels | Run once per candidate; never used for any choice |

The test file list is frozen with a hash in `model/data/splits.json`. The eval script refuses to
run if it changes.

---

## 6. Training

- **H1 (YOLO26n/s head, CrowdHuman):** 640 px, about 40 epochs, seed 0, all configs in
  `model/configs/`, every run logged to `model/runs/<name>/` (config, git hash, versions, data
  hash, metrics per epoch).
  - On this M4 Pro (MPS), I expect hours, not minutes, so **this needs your OK (over 1 hour)**.
  - Options: run overnight locally, or on Google Colab's free GPU with your existing Google
    account (no new sign-up).
  - I'll time a 1-epoch probe first (a few minutes) and give you the projected total before
    starting the full run.
- **H2 (RF-DETR Nano head):** stretch goal, same rules.
- **Fine-tuning on train interiors:** not planned. All our labelled train frames go to
  validation/test. With more labelled frames later (a third "train" split), this is the obvious
  next step, and the report says so.

---

## 7. Evaluation: one script, `make eval`

`model/aera_vision/eval.py` runs every candidate on the same frames and writes
`model/results/*.json` and `*.csv`, with charts in `model/results/charts/` (SVG/PNG drawn from
those files). The report and the app read only those files.

- **Detection (detectors only):**
  - **Box metrics on CrowdHuman val and the ~100 box-labelled train frames:** precision, recall
    and F1 at the chosen threshold and IoU 0.5; mAP@0.5; mAP@0.5:0.95; precision–recall curve.
    Head detectors are scored against head boxes; body detectors against full-body boxes.
  - **Point-based P/R/F1 on all train-interior test frames:** a prediction matches a head point
    one-to-one (Hungarian matching). A head box matches when the point lies inside it; a body box
    matches when the point lies in its top third.
  - **Threshold:** the F1-optimal confidence threshold is chosen on the **validation** split and
    then fixed. The report states the chosen value and shows the validation F1 curve.
- **Counting (every candidate):** MAE, RMSE, MAPE (frames with true count 0 excluded and counted
  separately), mean signed error (negative means undercounting), and a predicted-vs-true scatter.
- **Level (what riders see):**
  - Per-level precision, recall and F1 for Quiet … Packed, macro-F1, accuracy, and a 5×5
    confusion matrix.
  - Within-one-level accuracy.
  - Agreement between level(true count / C) and the eye-judged level (section 2).
- **By level:** all counting metrics broken down by true level.
- **Uncertainty:** 95 % bootstrap intervals (2,000 resamples, seed fixed, resampling by scene so
  frames from one video stay together) on every headline number.
- **Speed:**
  - Milliseconds per frame and FPS on this Mac: CPU, MPS, and ONNX Runtime on CPU (median of 200
    runs after 20 warm-ups, batch 1, at the deployment resolution).
  - Parameters and file size, read from the model files.
  - Pi 5 + AI HAT+ / AI HAT+ 2 (Hailo) and Jetson Orin Nano Super figures are **estimates quoted
    from named published benchmarks, labelled as not measured.**
- **Sensor confidence table:** σ per predicted level band (validation split) and the resulting
  `confidence` value the sim sends.
- **Reproducibility:** each results file records versions, seeds, data hashes, the command used
  and the date.

**Winner rule (fixed now, before seeing any numbers):**

1. Highest level macro-F1 on the **validation** split, among candidates expected to run at
   2 fps or more on a Pi 5 + AI HAT+ or a Jetson Orin Nano.
2. Tie (within the bootstrap interval): smaller |mean signed error| on Crowded + Packed.
3. Then the smaller model.
4. Then the more permissive licence.

The report then gives every candidate's **test** numbers and a short, honest justification,
including the winner's weaknesses.

---

## 8. Deliverables in `model/`

- `MODEL_REPORT.md`: method, datasets (sources, licences, sizes, splits), results tables for every
  candidate, charts, the chosen model, limitations, how to reproduce each number. The tables are
  generated from `results/` by a script, so no number is typed by hand.
- `MODEL_CARD.md`:
  - Intended use: counting people in a car, sending counts only.
  - Out of scope: identifying or tracking people, security use, staffing decisions on its own.
  - Privacy.
  - Known failures: occlusion, window reflections, people seen through doors, night, bags,
    prams, children.
- `results/*.json`, `results/*.csv`, `results/charts/`.
- `make eval` reruns everything from the frozen splits. Training has its own targets
  (`make train-head`).

---

## 9. The app (Part 2)

**Stack:**

- **Next.js 16 + TypeScript + Tailwind 4 + `@phosphor-icons/react`**, the same as AERA.
  Charts are hand-written SVG with a text table next to each (no chart library).
- **Hosted mode needs no GPU:** demo frames are stored already face-blurred, with the chosen
  model's precomputed outputs for each frame in `app/public/demo/`.
- **Live mode (local only):** a small FastAPI server (`model/aera_vision/serve.py`) runs the model
  on a webcam or video file. It blurs faces before any frame leaves it, and sends the browser a
  blurred preview plus counts.
- **One implementation of the device logic (smoother + sender), in TypeScript**, used by both
  modes and tested with AERA's own rules. The Python server only supplies per-frame counts.

**Screens** (bottom tabs on phones, top bar on a laptop):

1. **Train:**
   - One Innovia train as four joined cars (AERA's `TrainCars` look).
   - Each car shows: a small blurred camera tile, count, crowding tag (icon + label + colour),
     meter, the model's confidence, and device status (online/offline, last reading "12 s ago",
     queue size, 2 cameras).
   - A header with station, direction and time.
2. **Camera** (tap a car):
   - The blurred frame with head points or boxes drawn on it, and the count over time.
   - **"What leaves the car":** the exact `CarCount` JSON of the last reading, next to a note
     that images never leave the device.
   - The queue and send log.
3. **Scenario controls** (a panel on the Train screen):
   - Train: station + direction, or a trip.
   - Time of day.
   - Scenario: Quiet morning, Rush hour, Event surge, Packed car next to an empty one.
   - Play/pause, speed 1×/5×/20×, and "Go offline" to test the queue.
   - The Kelana Jaya Line has no stadium station. I'll model the event surge at **KLCC (KJ10)** and
     **Pasar Seni (KJ14, the nearest to Stadium Merdeka)** unless you prefer others.
   - Car loads over time come from AERA's crowding model for that stretch and hour (copied
     read-only from AERA's `data/models/KJ.json`, source noted). At each step the sim shows a test
     frame whose labelled count is close to the target, with the model's real output on that frame.
     So every number on screen is a real model result on a real frame.
4. **Model:** the credibility page.
   - Headline precision, recall, F1, MAE and level macro-F1 with 95 % intervals.
   - Confusion matrix, PR curve, error by level, speed, licences, limitations.
   - All of it read from `model/results/`, which is copied into the app at build time by a script.
5. **Send to AERA** (in Settings, **off by default**):
   - Backend URL + ingest token, kept in memory only and never stored.
   - When on, the sim picks a train that is running in that AERA instance and posts each reading
     to `/api/ingest/train/{device}`; the send log shows each reply.
   - When off, readings are only logged.
   - If AERA's CORS settings block the browser (AERA's deployed backend allows
     `aera-*.vercel.app` and its own dev origin), a small forwarder route in the app sends the
     reading instead.

**Design:**

- White background, Plus Jakarta Sans with tabular figures.
- Tokens copied from AERA's `theme.css` into one `app/styles/theme.css`, and no other colours.
- Cards with soft shadows and a 16 px radius, pill controls, Phosphor icons, no emoji.
- Crowding colour always with its icon and label.
- The Camera and Send log screens may use AERA's navy operator chrome (`--ops`) to mark them as
  technical.
- Express logo from `brand/logo/svg`, with "Vision" set next to it.
- Product voice: no jargon; a phrase after "·" or ":" starts with a capital; level names
  capitalised.
- Keyboard works throughout; contrast checked.
- Checked in a browser at 1440 px and 390 px.

---

## 10. Repo layout

```
PLAN.md  README.md  CLAUDE.md  Makefile          # make eval | make check | make app | make train-head
contracts/car-count.schema.json                   # copy of AERA's; a test checks it still matches
model/   pyproject.toml (uv, Python 3.12)
         aera_vision/  candidates/ data.py metrics.py levels.py eval.py bench.py report.py serve.py
         configs/  runs/ (git-ignored weights)  data/ (git-ignored images; manifests and splits kept)
         results/  MODEL_REPORT.md  MODEL_CARD.md  tests/
app/     Next.js + TS; lib/device (smoother, sender, contract), lib/crowd/levels.ts, components/, tests
```

## 11. Tests and `make check`

- **Contract:** every message the sim builds validates against `contracts/car-count.schema.json`.
  The schema copy equals AERA's file, checked when the sibling repo is present and skipped
  otherwise.
- **Levels:** thresholds and names match AERA's `levels.ts` (parsed), with boundary cases at 19.9,
  20, 94.9, 95, 100 and 100.1 %.
- **Metrics:** MAE, RMSE, MAPE, mean signed error, P/R/F1, IoU, mAP, macro-F1, the confusion matrix
  and bootstrap, each checked on a small example worked out by hand in the test.
- **Device logic:** the smoother window/min-change/30 s rule; the sender's `seq`, ordered flush,
  queue cap, the 120 s drop, retry on 5xx/429/offline, and no retry on 4xx. These mirror AERA's
  `edge/tests`.
- **`make check`:** ruff + pytest (model), eslint + `node --test` + `tsc --noEmit` (app). I run it
  before finishing.

## 12. What I can and can't measure on this Mac (M4 Pro, 24 GB, MPS)

| Can measure | Cannot measure (said so in the report) |
| --- | --- |
| Every accuracy metric on the train-interior val/test, CrowdHuman val and ShanghaiTech B test | Speed on a real Pi 5 + Hailo or Jetson: no hardware; Hailo's compiler runs only on x86 Linux. Published figures are quoted and labelled as estimates. |
| Speed on this Mac's CPU, MPS and ONNX Runtime CPU; model size | Accuracy from a real ceiling camera in an Innovia car, unless we record our own footage |
| Head-detector fine-tuning (slow on MPS; Colab is the fallback) | Night or tunnel lighting beyond what the collected images happen to contain |
| | Long-run drift, heat or camera-fogging effects of a real installation |

## 13. Order of work

1. **Now, no approval needed:**
   - Scaffold the repo, theme and contract.
   - Device logic and metric functions with tests.
   - Commons collection script (metadata and thumbnails only) and the labelling setup.
   - Download pretrained weights, all under 2 GB: YOLO11/26 n/s/m, RF-DETR N/S/M, ZIP, DM-Count.
     Then ShanghaiTech B for the reproduction check.
2. **After your OK:** CrowdHuman download → 1-epoch timing probe → head-detector training.
3. **You label** the train-interior set (val first, so thresholds can be chosen while the test set
   is labelled).
4. Full evaluation, `MODEL_REPORT.md`, model card, winner.
5. The app, wired to `model/results/` and the demo frames; browser check at 1440 / 390 px.
6. README with the pitch paragraph, `make check`, update CLAUDE.md.

## 14. Decisions I need from you

1. **Test footage:** can we record our own footage on Kelana Jaya trains (with Prasarana permission
   and a consent notice), or do we rely on public images and video only? Without our own footage
   the report will say it cannot show the real ceiling-camera angle.
2. **Labelling:** will you (or the group) label about 400 frames in Label Studio (about 6–9 hours)?
   I can't produce ground truth myself without it counting as made-up.
3. **CrowdHuman download** (over 2 GB; non-commercial research licence): OK to download train01 +
   val? I'll give the exact sizes before starting.
4. **Head-detector training** (likely more than 1 hour): overnight on this Mac, or on Colab with
   your Google account?
5. **View-capacity rule** (section 2): OK, or do you want strict count / 200 even for partial views?
6. **Two cameras per car** as the hardware assumption: OK?
7. **Event-surge stations:** KLCC and Pasar Seni?
8. **Name:** "AERA Vision" works well (it matches AERA's naming and says "camera"). Keep it?

## Sources

- Ultralytics YOLO26 release (14 Jan 2026): https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai
- Ultralytics licence: https://www.ultralytics.com/license and the repo's LICENSE (AGPL-3.0)
- RF-DETR docs and sizes: https://rfdetr.roboflow.com/latest/ and the repo LICENSE (Apache-2.0)
- ZIP (Ma et al., 2025): https://github.com/Yiming-M/ZIP (MIT), https://arxiv.org/abs/2506.19955
- DM-Count: https://github.com/cvlab-stonybrook/DM-Count (MIT)
- P2PNet licence (academic research only): https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- APGCC: https://github.com/AaronCIH/APGCC (MIT)
- CrowdHuman: http://www.crowdhuman.org/ (terms: non-commercial research and education)
- ShanghaiTech: Zhang et al., CVPR 2016; NWPU-Crowd: https://arxiv.org/abs/2001.03360
- RPEE-Heads (platform heads, not train interiors): https://arxiv.org/abs/2411.18164
- Pi AI HAT+ 2 review: https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/
- Pi AI HAT+ 2 (Jeff Geerling): https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2
- Rapid Rail crowd monitoring plan: https://www.lowyat.net/2025/366819/rapid-rail-to-introduce-crowd-monitoring-system-to-trains/
