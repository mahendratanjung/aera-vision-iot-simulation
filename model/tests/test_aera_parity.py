"""Our copies of AERA's rules must match AERA itself (read-only, when the sibling repo is present)."""

import json
import re
from pathlib import Path

import pytest

from aera_vision import levels

REPO = Path(__file__).resolve().parents[2]
AERA = REPO.parent / "aera-advanced ecosystem-of-rail-analytics"
needs_aera = pytest.mark.skipif(not AERA.exists(), reason="AERA repo not next to this one")


@needs_aera
def test_levels_match_aera_levels_ts():
    src = (AERA / "frontend/lib/crowd/levels.ts").read_text()
    rows = re.findall(r'\{ level: (\d), name: "([A-Za-z]+)", hint: "([^"]+)", from: (\d+) \}', src)
    assert [(int(a), b, c, float(d)) for a, b, c, d in rows] == levels.LEVELS
    bounds = re.search(r"DEFAULT_BOUNDS = \[([\d, ]+)\]", src).group(1)
    assert [float(x) for x in bounds.split(",")] == levels.BOUNDS


@needs_aera
def test_contract_copy_matches_aera():
    ours = json.loads((REPO / "contracts/car-count.schema.json").read_text())
    theirs = json.loads((AERA / "packages/contracts/car-count.schema.json").read_text())
    assert ours == theirs


@needs_aera
def test_camera_error_constants_match_aera():
    from aera_vision import evaluate

    fusion = (AERA / "backend/aera_twin/crowdsource/fusion.py").read_text()
    sensors = (AERA / "backend/aera_twin/crowdsource/sensors.py").read_text()
    assert f'"camera": {evaluate.AERA_CAMERA_SD}' in fusion
    assert f"max({evaluate.AERA_MIN_CONFIDENCE}, msg.confidence" in sensors
    assert "CARS = 4" in sensors


@pytest.mark.parametrize(("pct", "name"), [(0, "Quiet"), (19.9, "Quiet"), (20, "Comfortable"), (49.99, "Comfortable"),
                                           (50, "Busy"), (74.9, "Busy"), (75, "Crowded"), (94.9, "Crowded"),
                                           (95, "Packed"), (100, "Packed"), (180, "Packed")])
def test_level_boundaries(pct, name):
    assert levels.name_of(levels.level_of(pct)) == name


def test_view_level_uses_half_a_car():
    assert levels.VIEW_CAPACITY == 100
    assert levels.name_of(levels.view_level(19)) == "Quiet"
    assert levels.name_of(levels.view_level(20)) == "Comfortable"
    assert levels.name_of(levels.view_level(95)) == "Packed"
