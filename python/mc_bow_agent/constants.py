"""Verified physical + action-space constants for the MC 1.16.5 bow agent.

Sources: minecraft.wiki/Arrow, minecraft.wiki/Hitbox, OpenAI VPT lib/actions.py.
All values are Minecraft Java 1.16.5. Treat this file as the single source of
truth; the M0 physics-parity gate must assert the live game matches these.
"""

# --- Arrow physics (per game tick) ---
ARROW_DRAG = 0.99              # velocity *= 0.99 each tick (air); water = 0.6
ARROW_GRAVITY = 0.05          # velocity.y -= 0.05 each tick (applied after drag)
BOW_FULL_CHARGE_SPEED = 3.0   # blocks/tick at full draw
CROSSBOW_SPEED = 3.15
DISPENSER_ARROW_SPEED = 1.1
ARROW_HITBOX = 0.5            # arrow entity is 0.5 x 0.5

# Bow draw curve: vanilla BowItem.getPowerForTime
FULL_CHARGE_TICKS = 20       # 1.0 s to full power
MIN_FIRE_TICKS = 3           # below this the bow will not loose an arrow (f < 0.1)
BOW_INACCURACY_GAUSS = 0.0075  # per-axis gaussian spread at release (inaccuracy=1.0)

PLAYER_EYE_HEIGHT = 1.62

# --- VPT camera (mouse) discretization (agent.py ACTION_TRANSFORMER_KWARGS) ---
CAMERA_BINSIZE = 2
CAMERA_MAXVAL = 10           # max degrees/tick/axis; input is CLIPPED to +-10
CAMERA_MU = 10
CAMERA_QUANTIZATION = "mu_law"
CAMERA_N_BINS = 11           # (2*maxval//binsize)+1
CAMERA_ZERO_BIN = 5          # maxval//binsize -> the "no turn" bin
# Degree centre of each of the 11 bins (mu-law undiscretize of bins 0..10):
CAMERA_BIN_CENTERS_DEG = (-10.0, -5.81, -3.22, -1.61, -0.62, 0.0,
                          0.62, 1.61, 3.22, 5.81, 10.0)

# 360/2400 = 0.15 deg/pixel converts HUMAN raw mouse pixels -> degrees.
# The env 'camera' field is ALREADY in DEGREES. Do NOT multiply aimbot degree
# output by this when recording. Kept only to name-and-shame the silent bug.
CAMERA_SCALER_DEG_PER_PIXEL = 360.0 / 2400.0

# --- Hostile mob hitboxes (width, height, eye_height) in blocks, 1.16.5 ---
MOB_HITBOX = {
    "zombie":   (0.6, 1.95, 1.74),
    "skeleton": (0.6, 1.99, 1.74),
    "creeper":  (0.6, 1.7,  1.445),
    "spider":   (1.4, 0.9,  0.65),   # wide & short: lateral lead dominates
    "enderman": (0.6, 2.9,  2.55),
    "husk":     (0.6, 1.95, 1.74),
}


def mob_aim_point_offset(mob_type: str) -> float:
    """Vertical offset (blocks) from mob feet to the recommended aim point
    (hitbox centre = height/2). Unknown types fall back to zombie."""
    _w, h, _eye = MOB_HITBOX.get(mob_type, MOB_HITBOX["zombie"])
    return h * 0.5
