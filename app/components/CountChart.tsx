"use client";
// A car's count over the last hour of simulated time: what the device sent, and what the photos hold.
import { LineChart, REFERENCE, SERIES, TableView } from "@/components/charts";
import { clock } from "@/components/ui";
import type { HistoryPoint } from "@/lib/sim/engine";

export function CountChart({ points }: { points: HistoryPoint[] }) {
  if (points.length < 2) return <p className="text-sm text-ink-3">The chart fills in as the train runs. Press Play on the Train page.</p>;
  const sent = points.filter((p) => p.sent !== null).map((p) => [p.t, p.sent!] as [number, number]);
  const truth = points.map((p) => [p.t, p.truth] as [number, number]);
  const last = points[points.length - 1];
  return (
    <LineChart
      series={[
        { name: "Sent by the device", color: SERIES[0], points: sent },
        { name: "Labelled in the photos", color: REFERENCE, dashed: true, points: truth },
      ]}
      xLabel="Simulated time" yLabel="People" xFormat={(t) => clock(t, false)}
      summary={`From ${clock(points[0].t, false)} to ${clock(last.t, false)}. Last value sent ${last.sent ?? "none yet"}, labelled ${last.truth}.`}
      extra={<TableView caption="Count over time" head={["Time", "Sent", "Labelled", "Model this frame"]}
        rows={[...points].reverse().slice(0, 60).map((p) => [clock(p.t), p.sent ?? "", p.truth, p.model])} />}
    />
  );
}
