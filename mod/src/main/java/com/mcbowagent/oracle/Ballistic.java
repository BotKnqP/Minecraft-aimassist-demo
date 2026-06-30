package com.mcbowagent.oracle;

import com.mcbowagent.config.RecorderConfig;

/**
 * Tick-accurate Minecraft arrow simulator + ballistic pitch solver. Java port of
 * {@code python/mc_bow_agent/ballistic.py} — kept in sync so the oracle (privileged
 * aimbot, F10) and the vision runtime (Python) solve the same drop with the same
 * answer. Update order per tick (minecraft.wiki/Arrow): v *= drag; v.y -= gravity;
 * pos += v.
 *
 * Coordinate convention here mirrors the Python module:
 *   pitch DOWN is POSITIVE (Minecraft convention).
 *   yaw 0 looks along +Z; we always solve in the player-target horizontal plane
 *   so yaw drops out of the bisection (we model XZ as a single "horiz" axis).
 */
public final class Ballistic {

    private Ballistic() {}

    /**
     * Returns the arrow's y at the moment it first reaches horizDist in the XZ plane
     * (linearly interpolated between the bracketing ticks). NaN if it never reaches
     * that distance within maxTicks — caller treats that as "out of range".
     */
    public static double heightAtDistance(double speed, double pitchDeg, double horizDist, int maxTicks) {
        double pitchRad = Math.toRadians(pitchDeg);
        // initial velocity from look direction. Pitch DOWN is positive -> y-component is -sin(pitch)
        double cosP = Math.cos(pitchRad);
        double vx = 0.0;                    // we collapse XZ to one axis: vz is the radial speed
        double vy = -Math.sin(pitchRad) * speed;
        double vz = cosP * speed;
        double px = 0.0, py = 0.0, pz = 0.0;
        double prevHz = 0.0, prevY = 0.0;
        for (int t = 0; t < maxTicks; t++) {
            // tick update: drag, gravity, then move
            vx *= RecorderConfig.ARROW_DRAG;
            vy *= RecorderConfig.ARROW_DRAG;
            vz *= RecorderConfig.ARROW_DRAG;
            vy -= RecorderConfig.ARROW_GRAVITY;
            px += vx;
            py += vy;
            pz += vz;
            double hz = Math.hypot(px, pz);
            if (hz >= horizDist) {
                if (hz == prevHz) return py;
                double f = (horizDist - prevHz) / (hz - prevHz);
                return prevY + f * (py - prevY);
            }
            prevHz = hz;
            prevY = py;
        }
        return Double.NaN;
    }

    /**
     * Solve the pitch (deg, down-positive) that lands a freshly-loosed arrow at
     * (horizDist, heightDelta). Bisection on the lower (direct) arc — the lookup
     * function is monotonically decreasing in pitch. Returns the bracket midpoint
     * after `iters` halvings (~1e-13 deg precision in 60 iters; 20 is plenty).
     */
    public static double solvePitch(double speed, double horizDist, double heightDelta) {
        double lo = -45.0;   // very steep up
        double hi = 30.0;    // moderate down
        for (int i = 0; i < 40; i++) {
            double mid = (lo + hi) / 2.0;
            double y = heightAtDistance(speed, mid, horizDist, 400);
            // NaN -> aimed too far up, never reached horizDist -> need bigger pitch (more down)
            if (Double.isNaN(y) || y - heightDelta > 0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) / 2.0;
    }
}
