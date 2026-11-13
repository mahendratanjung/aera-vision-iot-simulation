# Model card: AERA Vision people counter

The chosen model, its measured accuracy and speed are in [MODEL_REPORT.md](MODEL_REPORT.md) and
[results/summary.json](results/summary.json). This card says what the counter is for, what it is not
for, and where it fails. It deliberately repeats no numbers: they live in the results, rewritten by
`make eval`.

## What it does

Counts the people in one camera view inside a train car, a few times a second, on a computer in the
car. The car's device adds its two cameras' counts and sends AERA one `CarCount` message: car number,
number of people, a confidence, and which train. Nothing else leaves the car.

## Intended use

- Estimating how full each car of a Kelana Jaya Line train is, for AERA's crowding levels (Quiet to
  Packed), fused with rider reports and the timetable model.
- A prototype and pitch demonstration (UTP Technopreneurship, Group 2). Not a deployed system.

## Out of scope

- Identifying, recognising or tracking any person, or keeping any image. The counter has no identity
  features and the design never stores or sends frames.
- Security, policing, enforcing women-only coaches, or any decision about an individual.
- Safety-critical decisions on its own (for example stopping boarding). Counts are estimates with a
  known error and must be combined with other sources.
- Commercial deployment as it stands: the CrowdHuman-trained parts and the test data are licensed for
  non-commercial research only, and Ultralytics/YOLOv5 code is AGPL-3.0.

## Privacy

- Frames are processed in memory on the device and dropped; only counts are sent (see the app's
  Camera page, "What leaves the car").
- Any preview shows faces blurred (the photos in the demo were blurred before they were saved; the
  live camera panel blurs the whole picture).
- The `CarCount` message holds no image, face, position or device owner. AERA replaces device ids with
  a keyed hash that changes daily.
- A real installation would still need Prasarana's approval, signage telling riders the cars have
  counting cameras, and a check against Malaysia's Personal Data Protection Act.

## Known failure cases

| Situation | What goes wrong | Mitigation |
| --- | --- | --- |
| Heavy occlusion in packed cars | People hidden behind others are missed; detectors undercount most when it matters most | Head detection and density counting (the candidates compared); report the error by level; pair with door counters or load-weigh |
| Reflections in windows and glass doors | Reflected people are counted twice | Region of interest that excludes windows; mount cameras so windows are behind them |
| People seen through open doors (platform) | Counted as passengers while the doors are open | Region of interest; count with doors closed (between stations) |
| Night and tunnel lighting | The Kelana Jaya Line runs underground from Ampang Park to Pasar Seni; low light or glare lowers recall | Not measured here: test photos are mostly daylight. Needs footage from a real car |
| Bags, prams, wheelchairs, coats on seats | Can be mistaken for people, or hide them | Not measured separately |
| Children | Small, often hidden; counted the same as adults, though a car's capacity in people assumes adults | Not measured separately |
| Fish-eye ceiling lenses | People near the edge are distorted | Test photos are not fish-eye; needs footage from the real camera |
| Two cameras' views overlap | The same person counted twice | Each camera counts only its half of the car (split region of interest) |

## Data

Measured on the CrowdHuman validation set (CC BY-NC 4.0), split once into tune and test. These are
crowds from the web, not train interiors; see the report's limitations.

## Maintenance

Rerun `make eval` after any change of model, threshold or data; the report, the app's Model page and
the device's confidence values all update from the same results file.
