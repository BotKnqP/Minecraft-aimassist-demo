package com.mcbowagent.oracle;

import com.mcbowagent.config.RecorderConfig;
import com.mcbowagent.vision.ProjectionUtil;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.Vec3d;

/**
 * v0 oracle: pick a target, aim straight at its hitbox centre, manage the bow.
 *
 * TODO(M0+): replace the straight-line aim with the ballistic lead+drop solver
 * (port python/mc_bow_agent/ballistic.py solve_pitch + lead_target). Do this only
 * AFTER the M0 parity gate confirms the per-tick arrow physics matches the game.
 */
public final class BowAimbotOracle {

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
        double[] ang = ProjectionUtil.lookAngles(eye, aim);   // {yaw, pitch}; TODO lead+drop

        boolean aligned = controller.stepView(p, ang[0], ang[1]);
        controller.manageBow(mc, aligned);
        return target.getEntityId();
    }
}
