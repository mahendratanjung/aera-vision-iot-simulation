// Crowding chip, meter, icon and the joined-car train: the same look as AERA's components/Crowd.tsx.
import { User, Users, UsersFour, UsersThree, Warning } from "@phosphor-icons/react/dist/ssr";
import { CROWD, type CrowdLevel, crowdInkVar, crowdOf, crowdSoftVar, crowdVar, pctShort } from "@/lib/crowd/levels";

/** One icon shape per level, so crowding never depends on colour alone. */
export function CrowdIcon({ level, size = 16, color }: { level: CrowdLevel; size?: number; color?: string }) {
  const props = { size, weight: "fill" as const, color: color ?? crowdInkVar(level), "aria-hidden": true };
  if (level === 1) return <User {...props} />;
  if (level === 2) return <Users {...props} />;
  if (level === 3) return <UsersThree {...props} />;
  if (level === 4) return <UsersFour {...props} />;
  return <Warning {...props} />;
}

/** Five segments filled up to the level. */
export function CrowdMeter({ level, size = "md" }: { level: CrowdLevel; size?: "sm" | "md" }) {
  const w = size === "sm" ? 6 : 9;
  const h = size === "sm" ? 6 : 8;
  return (
    <span aria-hidden className="inline-flex items-center gap-[3px]">
      {CROWD.map((s) => (
        <span key={s.level} style={{ width: w, height: h, background: s.level <= level ? crowdVar(level) : "var(--crowd-empty)" }}
          className="rounded-full" />
      ))}
    </span>
  );
}

/** The standard crowding chip: soft tinted pill, icon, level name and optional load. */
export function CrowdTag({ loadPct, showPct = false, size = "md", className = "" }: {
  loadPct: number; showPct?: boolean; size?: "sm" | "md" | "lg"; className?: string;
}) {
  const s = crowdOf(loadPct);
  const pad = size === "sm" ? "px-2 py-0.5 text-xs gap-1" : size === "lg" ? "px-3.5 py-1.5 text-[15px] gap-2" : "px-2.5 py-1 text-[13px] gap-1.5";
  return (
    <span className={`inline-flex items-center rounded-full font-semibold ${pad} ${className}`}
      style={{ background: crowdSoftVar(s.level), color: crowdInkVar(s.level) }}>
      <CrowdIcon level={s.level} size={size === "lg" ? 18 : size === "sm" ? 13 : 15} />
      {s.name}
      {showPct && <span className="num font-medium opacity-80">{pctShort(loadPct)}</span>}
    </span>
  );
}

/** A level by number (1..5), for tables and legends. */
export function LevelChip({ level, size = "sm" }: { level: CrowdLevel; size?: "sm" | "md" }) {
  return <CrowdTag loadPct={CROWD[level - 1].from + 0.1} size={size} />;
}

/**
 * The train as one row of four joined cars, front first, each filled with its crowding colour
 * and level icon, named underneath (AERA's TrainCars). Cars can be buttons here, to open a camera.
 */
export function TrainCars({ cars, selected, onSelect }: { cars: number[]; selected?: number; onSelect?: (car: number) => void }) {
  const lv = cars.map((c) => crowdOf(c));
  return (
    <div>
      <div className="flex h-12 gap-[3px]" role={onSelect ? "group" : "img"}
        aria-label={`Cars, front first: ${lv.map((x, i) => `car ${i + 1} ${x.name}`).join(", ")}`}>
        {lv.map((x, i) => {
          const style = {
            background: crowdVar(x.level),
            borderRadius: i === 0 ? "22px 6px 6px 22px" : i === cars.length - 1 ? "6px 12px 12px 6px" : 6,
            boxShadow: selected === i + 1 ? "0 0 0 2px var(--paper), 0 0 0 4px var(--ink)" : undefined,
          };
          const icon = <CrowdIcon level={x.level} size={20} color={x.level === 2 ? crowdInkVar(2) : "var(--paper)"} />;
          return onSelect ? (
            <button key={i} type="button" onClick={() => onSelect(i + 1)} style={style} aria-pressed={selected === i + 1}
              aria-label={`Car ${i + 1}: ${x.name}. Open its camera`}
              className="pressable flex flex-1 items-center justify-center">{icon}</button>
          ) : (
            <span key={i} style={style} className="flex flex-1 items-center justify-center">{icon}</span>
          );
        })}
      </div>
      <div aria-hidden className="mt-1.5 flex gap-[3px]">
        {lv.map((x, i) => (
          <span key={i} className="min-w-0 flex-1 truncate text-center text-[11px] leading-tight">
            <span className={`block font-semibold ${selected === i + 1 ? "text-ink" : "text-ink-3"}`}>{i === 0 ? "Front" : `Car ${i + 1}`}</span>
            <span className="block font-semibold" style={{ color: crowdInkVar(x.level) }}>{x.name}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
