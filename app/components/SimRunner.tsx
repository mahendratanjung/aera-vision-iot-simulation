"use client";
// Loads the photo bank and the measured error table, then runs the simulation clock 4 times a second.
import { useEffect } from "react";
import { useSim, type Bank } from "@/lib/store";
import calibration from "@/generated/calibration.json";
import type { Band } from "@/lib/device/confidence";

const STEP_S = 0.25;

export function SimRunner() {
  const setBank = useSim((s) => s.setBank);
  const step = useSim((s) => s.step);

  useEffect(() => {
    const cal = calibration as { missing?: boolean; bands?: Band[] };
    if (cal.missing || !cal.bands) {
      setBank(null, [], "The model has not been evaluated yet. Run make eval, then reload.");
      return;
    }
    fetch("/demo/bank.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((bank: Bank) => setBank(bank, cal.bands!))
      .catch(() => setBank(null, cal.bands!, "The camera photos are missing. Run make demo, then reload."));
  }, [setBank]);

  useEffect(() => {
    const id = window.setInterval(() => step(STEP_S), STEP_S * 1000);
    return () => window.clearInterval(id);
  }, [step]);

  return null;
}
