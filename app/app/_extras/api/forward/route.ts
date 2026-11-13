/**
 * Forwards a reading to the AERA backend the user typed in. Needed because AERA's backend only
 * accepts browser requests from AERA's own web address. Only three AERA paths are allowed, so this
 * cannot be used to send anything anywhere else. The token is passed through and never stored.
 */
import { NextResponse } from "next/server";
import { forwardTarget, MAX_BODY } from "@/lib/forward";

export async function POST(req: Request) {
  let input: { target?: string; path?: string; body?: string; token?: string; method?: string; query?: string };
  try {
    input = await req.json();
  } catch {
    return NextResponse.json({ error: "Bad request" }, { status: 400 });
  }
  const { target = "", path = "", body = "", token = "", method = "POST", query = "" } = input;
  if (body.length > MAX_BODY) return NextResponse.json({ error: "Reading too large" }, { status: 413 });
  const isGet = method === "GET";
  const t = forwardTarget(target, path, method, query);
  if ("error" in t) return NextResponse.json({ error: t.error }, { status: 400 });
  const url = t.url;
  try {
    const res = await fetch(url, {
      method: isGet ? "GET" : "POST",
      headers: isGet ? {} : { "Content-Type": "application/json", ...(token ? { "X-Ingest-Token": token } : {}) },
      body: isGet ? undefined : body,
      signal: AbortSignal.timeout(8000),
      cache: "no-store",
    });
    const text = await res.text();
    let parsed: unknown = null;
    try { parsed = JSON.parse(text); } catch { parsed = text.slice(0, 200); }
    return NextResponse.json({ status: res.status, body: parsed });
  } catch {
    return NextResponse.json({ error: "AERA did not answer" }, { status: 502 });
  }
}
