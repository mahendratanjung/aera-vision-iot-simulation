import assert from "node:assert/strict";
import { test } from "node:test";
import { forwardTarget } from "./forward.ts";

test("only AERA's sensor paths are forwarded", () => {
  const ok = forwardTarget("http://localhost:8000", "/api/ingest/train/cam-t07-c1", "POST");
  assert.ok("url" in ok && ok.url.href === "http://localhost:8000/api/ingest/train/cam-t07-c1");
  const sub = forwardTarget("https://aera.example.com/base", "/api/health", "GET");
  assert.ok("url" in sub && sub.url.href === "https://aera.example.com/api/health");
  const q = forwardTarget("http://x:8000", "/api/stations/KJ15/next-trains", "GET", "limit=1");
  assert.ok("url" in q && q.url.search === "?limit=1");
  for (const [path, method] of [["/api/admin/users", "GET"], ["/api/ingest/train/../admin", "POST"], ["/api/health", "POST"],
    ["/api/ingest/train/cam 1", "POST"], ["//evil.com/x", "POST"], ["/api/ingest/train/cam-1", "DELETE"]]) {
    assert.ok("error" in forwardTarget("http://localhost:8000", path, method), `${method} ${path} must be refused`);
  }
  assert.ok("error" in forwardTarget("file:///etc/passwd", "/api/health", "GET"));
  assert.ok("error" in forwardTarget("not a url", "/api/health", "GET"));
});
