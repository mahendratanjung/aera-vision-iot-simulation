"""Crowding levels, exactly as AERA defines them (frontend/lib/crowd/levels.ts).

Load is people as a share of capacity. A car holds 200 people (800 per four-car train).
AERA Vision assumes two cameras per car, one at each end, each counting its half of the car,
so one camera view holds VIEW_CAPACITY = 100 people at full load. A test image is treated as
one camera view: level = level_of(count / VIEW_CAPACITY * 100).
"""

from __future__ import annotations

CAR_CAPACITY = 200
CAMERAS_PER_CAR = 2
VIEW_CAPACITY = CAR_CAPACITY // CAMERAS_PER_CAR

# (level, name, hint, lower bound in % of capacity), same order and wording as levels.ts
LEVELS: list[tuple[int, str, str, float]] = [
    (1, "Quiet", "Plenty of seats", 0),
    (2, "Comfortable", "Room to stand", 20),
    (3, "Busy", "Standing room only", 50),
    (4, "Crowded", "Shoulder to shoulder", 75),
    (5, "Packed", "Boarding may not be possible", 95),
]
NAMES = [name for _, name, _, _ in LEVELS]
BOUNDS = [lo for *_, lo in LEVELS[1:]]  # [20, 50, 75, 95], AERA's DEFAULT_BOUNDS


def level_of(load_pct: float) -> int:
    """Crowding level 1..5 for a load in percent of capacity (AERA's crowdOf)."""
    for level, _, _, lo in reversed(LEVELS):
        if load_pct >= lo:
            return level
    return 1


def name_of(level: int) -> str:
    return NAMES[level - 1]


def view_level(count: float, capacity: float = VIEW_CAPACITY) -> int:
    """Level of one camera view holding ``count`` people."""
    return level_of(count / capacity * 100)
