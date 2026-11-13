/** Which AERA paths the app's forwarder may call (app/api/forward/route.ts). Nothing else is forwarded. */
export const POST_PATH = /^\/api\/ingest\/train\/[A-Za-z0-9_-]{1,64}$/;
export const GET_PATHS = [/^\/api\/health$/, /^\/api\/stations\/KJ\d{1,2}\/next-trains$/];
export const MAX_BODY = 4096; // AERA rejects bigger readings anyway

/** The URL to call, or a reason to refuse. */
export function forwardTarget(target: string, path: string, method: string, query = ""): { url: URL } | { error: string } {
  let base: URL;
  try {
    base = new URL(target);
  } catch {
    return { error: "The AERA address must start with http:// or https://" };
  }
  if (base.protocol !== "http:" && base.protocol !== "https:") return { error: "The AERA address must start with http:// or https://" };
  const ok = method === "GET" ? GET_PATHS.some((p) => p.test(path)) : method === "POST" && POST_PATH.test(path);
  if (!ok) return { error: "Only AERA's sensor paths can be used" };
  const url = new URL(path, base.href.replace(/\/?$/, "/"));
  if (method === "GET" && /^limit=\d{1,2}$/.test(query)) url.search = query;
  return { url };
}
