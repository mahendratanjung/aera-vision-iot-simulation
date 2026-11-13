"use client";
// The Model page body. A client component because its charts take formatter functions.
import { CheckCircle, Info, Warning } from "@phosphor-icons/react/dist/ssr";
import { GroupedBars, LineChart, Scatter, SERIES, TableView } from "@/components/charts";
import { LevelChip } from "@/components/Crowd";
import { Notice } from "@/components/ui";
import type { CrowdLevel } from "@/lib/crowd/levels";
import { results as R, type Candidate, type CI } from "@/lib/results";

const pct = (x: number | null | undefined, d = 1) => (x == null ? "n/a" : `${(x * 100).toFixed(d)}%`);
const n1 = (x: number | null | undefined, d = 1) => (x == null ? "n/a" : x.toFixed(d));
const signed = (x: number | null | undefined, d = 1) => (x == null ? "n/a" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`);
const ciPct = (c?: CI) => (c ? `${pct(c[0])} to ${pct(c[1])}` : "");
const ciNum = (c?: CI, d = 1) => (c ? `${n1(c[0], d)} to ${n1(c[1], d)}` : "");
const KIND: Record<Candidate["kind"], string> = { person: "Finds people", head: "Finds heads", density: "Counts without boxes", hybrid: "Hybrid" };

export function ModelView() {
  if (R.missing) {
    return <div className="mx-auto max-w-xl pt-10"><Notice tone="warn">No results yet. Run make eval in the project folder, then reload.</Notice></div>;
  }
  const byKey = Object.fromEntries(R.candidates.map((c) => [c.key, c]));
  const w = byKey[R.winner.key];
  const detPart: Candidate | undefined = w.kind === "hybrid" ? byKey[w.parts![0]] : w.kind === "density" ? undefined : w;
  const base = byKey["yolo11n"];
  const det = detPart?.test_detection;
  const t = w.test;
  const packed = t.by_level.find((b) => b.level === 5);
  const crowded = t.by_level.find((b) => b.level === 4);
  const worst = [...t.by_level].filter((b) => b.n > 0 && b.mae != null).sort((a, b) => b.mae! - a.mae!)[0];
  const others = R.candidates.filter((c) => c.key !== w.key);
  const detectors = R.candidates.filter((c) => c.test_detection);
  const bestOtherDet = detectors.filter((c) => c.key !== detPart?.key && c.key !== "yolo11n")
    .sort((a, b) => b.test_detection!.ap50 - a.test_detection!.ap50)[0];
  const prSeries = [detPart, base, bestOtherDet].filter((c, i, a): c is Candidate => !!c && a.findIndex((x) => x?.key === c.key) === i).slice(0, 3);
  const maxCount = Math.max(...(t.per_image?.true ?? [100]), ...(t.per_image?.pred ?? [100]));
  const levelsOrder = [1, 2, 3, 4, 5] as CrowdLevel[];

  return (
    <div className="grid gap-5">
      <header className="grid gap-1">
        <p className="label">Model · Measured, not claimed</p>
        <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">How well the camera counts</h1>
        <p className="max-w-3xl text-sm text-ink-2">
          Every number here was measured on <span className="num font-semibold">{R.dataset.test.toLocaleString()}</span> photos of crowds that no model was
          trained or tuned on ({R.dataset.name}). One photo stands in for one camera view, which holds {R.rules.view_capacity} people when full
          (two cameras per {R.rules.car_capacity}-person car). Ranges are 95% intervals from {R.rules.bootstrap.resamples.toLocaleString()} resamples.
        </p>
        {R.dataset.test_note && <p className="max-w-3xl text-xs text-ink-3">{R.dataset.test_note}</p>}
      </header>

      <section className="card-raised grid gap-4 p-4 sm:p-5" aria-labelledby="w-h">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="label">Chosen model</p>
            <h2 id="w-h" className="text-xl font-bold text-ink">{w.label}</h2>
            <p className="text-sm text-ink-2">{KIND[w.kind]} · {w.licence}</p>
          </div>
          <p className="max-w-md text-xs text-ink-3">Chosen by the rule fixed before any result was seen: best crowding-level score on the tuning
            photos{R.rules.speed_rule_applied ? `, among models that run at ${R.rules.speed_rule_cpu_fps} frames a second or more on this laptop’s processor alone` : ""}.
            The test photos were scored once, afterwards.{R.rules.speed_note ? ` ${R.rules.speed_note}` : ""}</p>
        </div>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Crowding level right" value={pct(t.levels.accuracy)} ci={ciPct(t.ci95?.level_accuracy)} />
          <Stat label="Level score (macro-F1)" value={n1(t.levels.macro_f1, 2)} ci={ciNum(t.ci95?.level_macro_f1, 2)} />
          <Stat label="Average miscount" value={`${n1(t.counting.mae)} people`} ci={ciNum(t.ci95?.mae)} />
          <Stat label="Precision" value={det ? pct(det.precision) : "n/a"} ci={ciPct(det?.ci95?.precision)} />
          <Stat label="Recall" value={det ? pct(det.recall) : "n/a"} ci={ciPct(det?.ci95?.recall)} />
          <Stat label="F1" value={det ? pct(det.f1) : "n/a"} ci={ciPct(det?.ci95?.f1)} />
        </dl>
        {w.kind === "hybrid" && detPart && <p className="text-xs text-ink-3">Precision, recall and F1 are for the part that finds {detPart.kind === "head" ? "heads" : "people"} ({detPart.label}),
          at its score threshold of {n1(detPart.threshold, 2)}, matched at overlap {R.rules.iou}. Above {w.switch} people in a view, the density counter takes over.</p>}
        <ul className="grid gap-1.5 text-sm text-ink-2">
          <li className="flex gap-2"><CheckCircle size={18} weight="fill" className="mt-0.5 shrink-0 text-good" aria-hidden />
            Against today&rsquo;s camera model in AERA ({base.label}): average miscount {n1(t.counting.mae)} vs {n1(base.test.counting.mae)} people,
            level score {n1(t.levels.macro_f1, 2)} vs {n1(base.test.levels.macro_f1, 2)}.</li>
          <li className="flex gap-2"><Warning size={18} weight="fill" className="mt-0.5 shrink-0 text-serious" aria-hidden />
            Hardest: {worst?.name} views, off by {n1(worst?.mae)} people on average.
            {crowded && packed && ` In Crowded and Packed views it is off by ${signed(crowded.mean_signed_error)} and ${signed(packed.mean_signed_error)} people on average (minus means it undercounts).`}</li>
          <li className="flex gap-2"><Info size={18} weight="fill" className="mt-0.5 shrink-0 text-accent" aria-hidden />
            Few test photos are Crowded ({R.dataset.test_levels.Crowded}) or Packed ({R.dataset.test_levels.Packed}), so those numbers are the least certain.</li>
        </ul>
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="card grid content-start gap-3 p-4" aria-labelledby="cm-h">
          <h2 id="cm-h" className="text-base font-bold text-ink">Crowding level: right and wrong</h2>
          <p className="text-xs text-ink-3">Rows are the level in the labels, columns the level from the model&rsquo;s count. The diagonal is right.</p>
          <Confusion m={t.levels.confusion} />
          <table className="num w-full text-left text-xs">
            <caption className="sr-only">Per level precision, recall and F1</caption>
            <thead><tr className="text-ink-3"><th className="py-1 font-semibold">Level</th><th className="font-semibold">Photos</th><th className="font-semibold">Precision</th><th className="font-semibold">Recall</th><th className="font-semibold">F1</th></tr></thead>
            <tbody>{t.levels.per_level.map((p) => (
              <tr key={p.level} className="border-t border-rule"><td className="py-1"><LevelChip level={p.level as CrowdLevel} /></td><td>{p.support}</td><td>{pct(p.precision, 0)}</td><td>{pct(p.recall, 0)}</td><td>{pct(p.f1, 0)}</td></tr>
            ))}</tbody>
          </table>
        </section>

        <section className="card grid content-start gap-3 p-4" aria-labelledby="lv-h">
          <h2 id="lv-h" className="text-base font-bold text-ink">Error as cars fill</h2>
          <GroupedBars
            groups={levelsOrder.map((l) => ({ key: String(l), label: <LevelChip level={l} /> }))}
            series={[
              { name: w.label, color: SERIES[0], values: levelsOrder.map((l) => t.by_level[l - 1].mae) },
              { name: `${base.label} (AERA today)`, color: SERIES[1], values: levelsOrder.map((l) => base.test.by_level[l - 1].mae) },
            ]}
            format={(v) => n1(v)} summary="Average miscount in people per camera view, by the level in the labels." />
          <h3 className="text-sm font-bold text-ink">Over or under?</h3>
          <GroupedBars zeroCentred
            groups={levelsOrder.map((l) => ({ key: String(l), label: <LevelChip level={l} /> }))}
            series={[
              { name: w.label, color: SERIES[0], values: levelsOrder.map((l) => t.by_level[l - 1].mean_signed_error) },
              { name: `${base.label} (AERA today)`, color: SERIES[1], values: levelsOrder.map((l) => base.test.by_level[l - 1].mean_signed_error) },
            ]}
            format={(v) => signed(v)} summary="Average signed error in people: left of the line undercounts, right overcounts." />
          <TableView caption="Error by level" head={["Level", "Photos", "Average miscount", "Over or under", `${base.label} miscount`]}
            rows={t.by_level.map((b, i) => [b.name, b.n, n1(b.mae), signed(b.mean_signed_error), n1(base.test.by_level[i].mae)])} />
        </section>

        {t.per_image && (
          <section className="card grid content-start gap-3 p-4" aria-labelledby="sc-h">
            <h2 id="sc-h" className="text-base font-bold text-ink">Counted against labelled</h2>
            <Scatter points={t.per_image.true.map((x, i) => [x, t.per_image!.pred[i]] as [number, number])} max={Math.min(maxCount, 250)}
              xLabel="People in the labels" yLabel="People counted"
              summary={`One dot per test photo (${t.counting.n.toLocaleString()}). Dots below the dashed line are undercounts. Average signed error ${signed(t.counting.mean_signed_error)} people.`} />
          </section>
        )}

        {prSeries.length > 0 && (
          <section className="card grid content-start gap-3 p-4" aria-labelledby="pr-h">
            <h2 id="pr-h" className="text-base font-bold text-ink">Finding people: precision and recall</h2>
            <LineChart xDomain={[0, 1]} yDomain={[0, 1]} xLabel="Recall (share of people found)" yLabel="Precision"
              xFormat={(v) => pct(v, 0)} yFormat={(v) => pct(v, 0)}
              series={prSeries.map((c, i) => ({ name: c.label, color: SERIES[i], points: c.test_detection!.pr_curve.recall.map((r, j) => [r, c.test_detection!.pr_curve.precision[j]] as [number, number]) }))}
              summary={`As the score threshold falls, recall rises and precision drops. Boxes count as right at overlap ${R.rules.iou} or more. ${prSeries.map((c) => `${c.label}: AP ${pct(c.test_detection!.ap50)}`).join("; ")}.`} />
          </section>
        )}
      </div>

      <section className="card grid content-start gap-3 overflow-hidden p-4" aria-labelledby="all-h">
        <h2 id="all-h" className="text-base font-bold text-ink">Every model tried</h2>
        <p className="text-xs text-ink-3">All used as published, without retraining. Test photos. Precision, recall and F1 at the threshold chosen on the tuning photos; AP is average precision at overlap 0.5, and over 0.5 to 0.95.</p>
        <div className="-mx-4 overflow-x-auto px-4">
          <table className="num w-full min-w-[1180px] whitespace-nowrap text-left text-xs">
            <caption className="sr-only">All candidate models on the test photos</caption>
            <thead><tr className="text-ink-3">
              {["Model", "Type", "Licence", "Params (M)", "Size (MB)", "CPU fps", "Precision", "Recall", "F1", "AP 0.5", "AP 0.5:0.95", "Miscount", "Over/under", "Level score", "Level right"].map((h) => <th key={h} className="py-1.5 pr-3 font-semibold">{h}</th>)}
            </tr></thead>
            <tbody>
              {[w, ...others].map((c) => {
                const d = c.test_detection;
                return (
                  <tr key={c.key} className={`border-t border-rule ${c.key === w.key ? "bg-accent-soft font-semibold" : ""}`}>
                    <td className="py-1.5 pr-3 font-sans text-ink">{c.label}</td>
                    <td className="pr-3 font-sans text-ink-2">{KIND[c.kind]}</td>
                    <td className="pr-3 font-sans text-ink-2">{c.licence.split(" (")[0]}</td>
                    <td className="pr-3">{n1(c.params_m)}</td><td className="pr-3">{n1(c.file_mb, 0)}</td><td className="pr-3">{n1(c.cpu_fps)}</td>
                    <td className="pr-3">{pct(d?.precision)}</td><td className="pr-3">{pct(d?.recall)}</td><td className="pr-3">{pct(d?.f1)}</td>
                    <td className="pr-3">{pct(d?.ap50)}</td><td className="pr-3">{pct(d?.ap50_95)}</td>
                    <td className="pr-3">{n1(c.test.counting.mae)} <span className="text-ink-3">({ciNum(c.test.ci95?.mae)})</span></td>
                    <td className="pr-3">{signed(c.test.counting.mean_signed_error)}</td>
                    <td className="pr-3">{n1(c.test.levels.macro_f1, 2)} <span className="text-ink-3">({ciNum(c.test.ci95?.level_macro_f1, 2)})</span></td>
                    <td className="pr-3">{pct(c.test.levels.accuracy)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="card grid content-start gap-3 p-4" aria-labelledby="sp-h">
          <h2 id="sp-h" className="text-base font-bold text-ink">Speed</h2>
          {R.speed ? (
            <>
              <p className="text-xs text-ink-3">Measured on this laptop ({R.speed.machine.chip}), one {R.speed.frame.size.join(" × ")} frame, median of 50 runs.
                The camera needs 2 frames a second.</p>
              <table className="num w-full text-left text-xs">
                <caption className="sr-only">Speed per model</caption>
                <thead><tr className="text-ink-3"><th className="py-1 font-semibold">Model</th><th className="font-semibold">Processor (ms)</th><th className="font-semibold">Graphics (ms)</th></tr></thead>
                <tbody>{R.candidates.filter((c) => R.speed!.candidates[c.key]).map((c) => {
                  const s = R.speed!.candidates[c.key];
                  return <tr key={c.key} className="border-t border-rule"><td className="py-1 font-sans">{c.label}</td><td>{n1(s.cpu?.ms_median, 0)}</td><td>{s.mps ? n1(s.mps.ms_median, 0) : "n/a"}</td></tr>;
                })}</tbody>
              </table>
            </>
          ) : <p className="text-sm text-ink-3">{R.rules.speed_note ?? "Speed not measured."}</p>}
          {R.edge && (
            <div className="grid gap-1.5 rounded-sign bg-surface-2 p-3 text-xs text-ink-2">
              <p className="font-semibold text-ink">On the train&rsquo;s computer (not measured here)</p>
              <p>{R.edge.note}</p>
              <ul className="grid gap-1">{R.edge.devices.map((d) => <li key={d.url + d.device}><span className="font-semibold text-ink">{d.device}:</span> {d.figure}. <a className="text-accent underline-offset-2 hover:underline" href={d.url}>{d.source}</a></li>)}</ul>
              <p>{R.edge.needed}</p>
            </div>
          )}
        </section>

        <section className="card grid content-start gap-3 p-4" aria-labelledby="cal-h">
          <h2 id="cal-h" className="text-base font-bold text-ink">How much AERA should trust each reading</h2>
          <p className="text-xs text-ink-3">AERA weighs a camera reading by its expected error. The device sends a confidence that turns into exactly
            the error measured here (on the tuning photos), by the level each camera&rsquo;s count falls in.</p>
          <table className="num w-full text-left text-xs">
            <caption className="sr-only">Measured error and confidence by level</caption>
            <thead><tr className="text-ink-3"><th className="py-1 font-semibold">Level counted</th><th className="font-semibold">Photos</th><th className="font-semibold">Error (people)</th><th className="font-semibold">Car error (points)</th><th className="font-semibold">Confidence</th></tr></thead>
            <tbody>{R.calibration.bands.map((b) => (
              <tr key={b.level} className="border-t border-rule"><td className="py-1"><LevelChip level={b.level as CrowdLevel} /></td><td>{b.n}</td>
                <td>{n1(b.rmse_used)}{b.borrowed ? "*" : ""}</td><td>{n1(b.car_sd_points_if_both_halves)}</td><td>{n1(b.confidence_if_both_halves, 2)}</td></tr>
            ))}</tbody>
          </table>
          <p className="text-xs text-ink-3">Car error assumes both cameras see the same level. AERA trusts a camera most at confidence 1 (error 8 points) and never below 0.3.
            {R.calibration.bands.some((b) => b.borrowed) && " * Fewer than 10 tuning photos: uses the larger error of this level and the one below."}</p>
        </section>
      </div>

      <section className="card grid gap-2 p-4 text-sm text-ink-2" aria-labelledby="lim-h">
        <h2 id="lim-h" className="text-base font-bold text-ink">Limits of this test</h2>
        <ul className="grid list-disc gap-1 pl-5">
          <li>The photos are general crowds from the web, not the inside of a Kelana Jaya train. A ceiling camera in a real car sees people from above and closer, with poles, windows and doors in view.</li>
          <li>One photo is treated as one camera view holding {R.rules.view_capacity} people. The level of a real car depends on how much of it each camera sees.</li>
          <li>Speed was not measured, neither on this laptop nor on the train&rsquo;s computer (Raspberry Pi 5 with an AI HAT, or Jetson); published figures for similar models are quoted above.</li>
          {R.dataset.test_note && <li>{R.dataset.test_note}</li>}
          <li>No model was retrained. The CrowdHuman head model was trained on CrowdHuman&rsquo;s training photos and may have used these validation photos to pick its checkpoint.</li>
          <li>CrowdHuman and the CrowdHuman-trained weights are for non-commercial research. Ultralytics models are AGPL-3.0.</li>
        </ul>
        <p className="text-xs text-ink-3">Data: <a className="text-accent" href={R.dataset.url}>{R.dataset.name}</a>, {R.dataset.licence}. Split seed {R.dataset.split_seed},
          {" "}{R.dataset.tune.toLocaleString()} tuning and {R.dataset.test.toLocaleString()} test photos. Generated {R.generated} by: {R.command}.</p>
      </section>
    </div>
  );
}

function Stat({ label, value, ci }: { label: string; value: string; ci: string }) {
  return (
    <div className="rounded-sign bg-surface-2 p-3">
      <dt className="text-xs font-semibold text-ink-3">{label}</dt>
      <dd className="num text-xl font-bold text-ink">{value}</dd>
      {ci && <dd className="num text-[11px] text-ink-3">95%: {ci}</dd>}
    </div>
  );
}

function Confusion({ m }: { m: number[][] }) {
  const names = ["Quiet", "Comfortable", "Busy", "Crowded", "Packed"];
  return (
    <div className="overflow-x-auto">
      <table className="num w-full min-w-[420px] border-separate border-spacing-[2px] text-center text-xs">
        <caption className="sr-only">Confusion matrix: rows labelled level, columns counted level</caption>
        <thead><tr><th className="text-left font-semibold text-ink-3">Labelled ↓ Counted →</th>{names.map((n) => <th key={n} className="font-semibold text-ink-3">{n}</th>)}</tr></thead>
        <tbody>{m.map((row, i) => {
          const tot = row.reduce((a, b) => a + b, 0) || 1;
          return (
            <tr key={i}>
              <th className="text-left font-semibold text-ink-2">{names[i]}</th>
              {row.map((v, j) => {
                const share = v / tot;
                return (
                  <td key={j} className="rounded-[4px] py-2" title={`${names[i]} labelled, ${names[j]} counted: ${v} photos (${Math.round(share * 100)}% of the row)`}
                    style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(share * 60)}%, var(--surface-2))`, color: "var(--ink)", fontWeight: i === j ? 700 : 500 }}>
                    {v}
                  </td>
                );
              })}
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}
