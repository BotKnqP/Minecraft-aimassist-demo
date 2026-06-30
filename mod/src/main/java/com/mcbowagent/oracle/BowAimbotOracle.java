package com.mcbowagent.oracle;

import com.mcbowagent.config.RecorderConfig;
import com.mcbowagent.vision.ProjectionUtil;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.Vec3d;

/**
 * Privileged oracle (F10): pick a target, solve the ballistic pitch (drop compensation),
 * aim at the lifted point, manage the bow. Same arrow physics as the Python runtime —
 * both call into the same solver ({@link Ballistic#solvePitch} / ballistic.solve_pitch).
 *
 * The straight-line aim from {@link ProjectionUtil#lookAngles} IGNORES gravity, so far
 * shots fall short (this was the "远处的僵尸打不到" symptom). With the solver in place,
 * the oracle's pitch is the angle that lands an arrow exactly at the hitbox centre at
 * the target's range, given the bow's full-charge speed.
 *
 * TODO(later): lead a moving target (port lead_target). Stationary aim_botz targets do
 * not need it; mode_natural zombies close radially so the lateral bias is small.
 */
public final class BowAimbotOracle {

    private static final double MAX_RANGE_BLOCKS = 40.0;  // beyond this the bow can't reach -> don't fire

    private final TargetSelector selector = new TargetSelector();
    private final BowMacroController controller = new BowMacroController();

    /** Runs one tick of the aimbot. Returns the engaged target's entity id, or -1. */
    public int tick(MinecraftClient mc, RecorderConfig cfg) {
        ClientPlayerEntity p = mc.player;
        if (p == null) return -1;

        Entity target = selector.select(p, mc.world, cfg);
        if (target == null) {
            controller.releaseBow(mc);
            return -1;
        }

        Vec3d eye = p.getCameraPosVec(1.0F);
        Vec3d aim = new Vec3d(target.getX(),
                target.getY() + target.getHeight() * 0.5, target.getZ());
        double[] ang = ProjectionUtil.lookAngles(eye, aim);   // straight-line {yaw, pitch}

        // Drop compensation: pitchSolved = the pitch (down-positive) at which a freshly-loosed full-charge
        // arrow lands at (horizDist, heightDelta). Replaces the straight-line pitch, which ignored gravity.
        double dx = aim.x - eye.x;
        double dz = aim.z - eye.z;
        double horiz = Math.hypot(dx, dz);
        double heightDelta = aim.y - eye.y;
        boolean inRange = horiz > 0.0 && horiz <= MAX_RANGE_BLOCKS;
        if (inRange) {
            ang[1] = Ballistic.solvePitch(RecorderConfig.BOW_FULL_CHARGE_SPEED, horiz, heightDelta);
        }

        boolean aligned = controller.stepView(p, ang[0], ang[1]);
        // Only release the bow when the target is actually reachable: track far targets but don't shoot
        // arrows at them (they'd fall short and waste ammo).
        if (inRange) {
            controller.manageBow(mc, aligned);
        } else {
            controller.releaseBow(mc);
        }
        return target.getEntityId();
    }
}
