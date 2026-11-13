"""Parsing CrowdHuman, the counting rule, and the choices made on the tune split."""

import numpy as np
import pytest

from aera_vision import candidates, data, evaluate


def _frame(vbox, zones, w=100, h=100):
    f = data.Frame("x", w, h, np.asarray(vbox, float).reshape(-1, 4), np.asarray(vbox, float).reshape(-1, 4),
                   np.zeros((0, 4)), np.asarray(zones, float).reshape(-1, 4), np.zeros((0, 4)))
    from aera_vision.metrics import centres_inside
    f.counted = ~centres_inside(f.vbox, f.zones)
    return f


def test_parse_marks_zones_and_ignored_people(tmp_path, monkeypatch):
    from PIL import Image

    Image.new("RGB", (50, 40)).save(tmp_path / "a.jpg")
    monkeypatch.setattr(data, "IMAGES", tmp_path)
    row = {"ID": "a", "gtboxes": [
        {"tag": "person", "fbox": [0, 0, 10, 20], "vbox": [0, 0, 10, 15], "hbox": [2, 0, 4, 4],
         "extra": {"ignore": 0}, "head_attr": {"ignore": 0}},
        {"tag": "person", "fbox": [20, 0, 10, 20], "vbox": [20, 0, 10, 20], "hbox": [22, 0, 4, 4],
         "extra": {"ignore": 1}},
        {"tag": "mask", "fbox": [40, 0, 10, 10], "vbox": [40, 0, 10, 10], "hbox": [40, 0, 10, 10], "extra": {"ignore": 1}},
    ]}
    f = data._parse(row)
    assert (f.width, f.height) == (50, 40)
    assert f.vbox.tolist() == [[0, 0, 10, 15]]  # x, y, w, h -> x1, y1, x2, y2
    assert f.zones.tolist() == [[20, 0, 30, 20], [40, 0, 50, 10]]
    assert f.head_ignore.tolist() == [[22, 0, 26, 4]]
    assert f.count == 1


def test_people_in_zones_are_not_counted_for_truth_or_detectors():
    f = _frame([[0, 0, 10, 10], [50, 50, 60, 60]], [[45, 45, 70, 70]])
    assert f.count == 1
    out = candidates.Output(np.array([[0, 0, 10, 10], [50, 50, 60, 60], [80, 0, 90, 10]], np.float32),
                            np.array([0.9, 0.9, 0.2], np.float32))
    assert evaluate.detector_count(out, f, 0.5) == 1  # in-zone box dropped, low score dropped
    assert evaluate.detector_count(out, f, 0.1) == 2


def test_density_in_zones_is_dropped():
    f = _frame([], [[0, 0, 50, 100]], w=100, h=100)
    den = np.ones((2, 2), np.float32)  # cell centres at 25 and 75
    assert evaluate.density_count(candidates.Output(density=den, cell=50), f) == pytest.approx(2.0)


def test_hybrid_switch_is_chosen_on_tune():
    # detector is right when sparse, undercounts when packed; the counter is the other way round
    true = np.array([5, 10, 15, 80, 90, 100], float)
    det = np.array([5, 10, 15, 40, 45, 50], float)
    den = np.array([12, 18, 25, 80, 90, 100], float)
    switch, fit = evaluate.choose_hybrid_switch(det, den, true)
    assert 16 <= switch <= 40
    assert fit["mae"] == 0


def test_confidence_follows_aera_rule():
    pred = np.array([10.0] * 20 + [60.0] * 20)
    true = pred - np.array([2.0] * 20 + [8.0] * 20)
    c = evaluate.calibrate_confidence(pred, true)
    quiet, busy = c["bands"][0], c["bands"][2]
    assert quiet["rmse_people"] == pytest.approx(2) and busy["rmse_people"] == pytest.approx(8)
    # car error with both halves Quiet: sqrt(2) * 2 / 200 * 100 = 1.41 points -> 8 / 1.41 > 1 -> 1.0
    assert quiet["confidence_if_both_halves"] == 1.0
    # Busy: sqrt(2) * 8 / 2 = 5.66 points -> 8 / 5.66 > 1 -> 1.0 as well
    assert busy["confidence_if_both_halves"] == 1.0
    big = evaluate.calibrate_confidence(np.array([60.0] * 20), np.array([30.0] * 20))["bands"][2]
    # error 30 people: sqrt(2) * 30 / 2 = 21.2 points -> 8 / 21.2 = 0.377
    assert big["confidence_if_both_halves"] == pytest.approx(8 / (np.sqrt(2) * 30 / 2))


def test_winner_rule():
    def row(key, f1, bias, params, fps=10.0, lic="MIT"):
        return {"key": key, "label": key, "licence": lic, "params_m": params, "cpu_fps": fps,
                "tune": {"levels": {"macro_f1": f1}, "crowded_packed_mean_signed_error": bias}}
    rows = [row("fast_ok", 0.70, -10, 5), row("tied_better_bias", 0.695, -2, 20),
            row("best_but_slow", 0.80, 0, 5, fps=1.0), row("worse", 0.5, 0, 1)]
    w = evaluate.choose_winner(rows)
    assert w["key"] == "tied_better_bias"
    assert w["excluded_too_slow"] == ["best_but_slow"]


def test_frozen_split_is_intact():
    if not data.SPLITS.exists():
        pytest.skip("no split yet")
    s = data.load_splits()
    assert len(s["tune"]) + len(s["test"]) == 4370
    assert not set(s["tune"]) & set(s["test"])
