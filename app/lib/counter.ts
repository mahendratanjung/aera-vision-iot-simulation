/**
 * Where the people counter runs. Locally (npm run dev) it is the counter on this computer; on the
 * deployed site, NEXT_PUBLIC_COUNTER_URL points at the counter on Render.
 */
export const COUNTER_URL = (process.env.NEXT_PUBLIC_COUNTER_URL || "http://localhost:8765").replace(/\/+$/, "");
export const COUNTER_IS_LOCAL = /^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(COUNTER_URL);

/** Plain words for "the counter did not answer", for each set-up. */
export const COUNTER_DOWN = COUNTER_IS_LOCAL
  ? "The counter is not running. Stop and run npm run dev again, then wait for “Counter ready”."
  : "The counter did not answer. It may still be waking up: try again in a minute.";

/** Shown while the first request may be waking a sleeping free server. */
export const COUNTER_WAKING = COUNTER_IS_LOCAL ? "Counting…" : "Counting… The first count can take up to a minute while the server wakes up.";

/** Where images go, in plain words. */
export const COUNTER_PRIVACY = COUNTER_IS_LOCAL
  ? "Your photo is counted on this computer and is not saved or sent anywhere."
  : "Your photo is sent to the AERA Vision counter, counted and deleted straight away. It is never saved.";

export async function countImage(blob: Blob, timeoutMs = 90_000) {
  const res = await fetch(`${COUNTER_URL}/count`, {
    method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" }, signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}
