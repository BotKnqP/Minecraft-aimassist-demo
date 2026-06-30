"""Per-tick expert-recording schema.

One demonstration yields THREE label sets every tick (visual / privileged truth
/ expert action) plus events. This dataclass is the CONTRACT between the Fabric
recorder (Java) and the Python training pipeline; keep both sides in sync.
"""
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple
import json


@dataclass
class MobTruth:
    type: str
    entity_id: int
    world_xyz: Tuple[float, float, float]   # feet position
    velocity: Tuple[float, float, float]    # blocks/tick
    health: float
    rel_yaw: float                          # deg, relative to player look
    rel_pitch: float
    distance: float
    radial_speed: float                     # closing speed (+ = approaching)
    tangential_speed: float
    visible: bool                           # in frustum AND centre ray not block-occluded (edges may clip)
    screen_bbox: Optional[Tuple[int, int, int, int]]  # (x0,y0,x1,y1) px or None


@dataclass
class ExpertAction:
    camera: Tuple[float, float] = (0.0, 0.0)  # (d_pitch, d_yaw) deg, per-tick relative
    forward: int = 0
    back: int = 0
    left: int = 0
    right: int = 0
    jump: int = 0
    sprint: int = 0
    sneak: int = 0
    use: int = 0                  # bow draw held (1) / released (0)
    hotbar: int = 0               # selected slot 1..9, 0 = unchanged
    target_entity_id: int = -1    # mob the aimbot is engaging (-1 = none)


@dataclass
class Events:
    arrow_released: bool = False
    arrow_hit: bool = False
    kill: bool = False
    damage_taken: bool = False


@dataclass
class TickRecord:
    tick: int
    frame_path: str               # PNG/JPEG relative path
    player_xyz: Tuple[float, float, float]
    player_yaw: float
    player_pitch: float
    health: float
    arrows: int
    bow_charge_ticks: int
    mobs: List[MobTruth] = field(default_factory=list)
    action: ExpertAction = field(default_factory=ExpertAction)
    events: Events = field(default_factory=Events)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_episode(jsonl_path: str):
    """Yield TickRecord dicts from an episode .jsonl file (one record per line)."""
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
