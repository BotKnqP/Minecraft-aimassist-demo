package com.mcbowagent.oracle;

import com.mcbowagent.config.RecorderConfig;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.mob.HostileEntity;

/** v0: nearest live hostile within scan radius. TODO: threat-score (creeper >> spider
 *  > zombie > skeleton-at-range) + target-commitment hysteresis. */
public final class TargetSelector {

    public Entity select(ClientPlayerEntity player, ClientWorld world, RecorderConfig cfg) {
        if (world == null) return null;
        Entity best = null;
        double bestDist = cfg.scanRadius * cfg.scanRadius;
        for (Entity e : world.getEntities()) {
            if (!(e instanceof HostileEntity)) continue;
            if (!e.isAlive()) continue;
            double d = player.squaredDistanceTo(e);
            if (d < bestDist) {
                bestDist = d;
                best = e;
            }
        }
        return best;
    }
}
