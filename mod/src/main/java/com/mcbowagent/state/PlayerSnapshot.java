package com.mcbowagent.state;

import com.mcbowagent.record.JsonUtil;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.item.Items;

/**
 * Top-level player fields of a TickRecord. writeFields() emits a JSON FRAGMENT
 * (key:value pairs, no surrounding braces) because these live at the record root.
 */
public final class PlayerSnapshot {

    public final double x, y, z;
    public final float yaw, pitch;
    public final float health;
    public final int arrows;
    public final int bowChargeTicks;

    private PlayerSnapshot(double x, double y, double z, float yaw, float pitch,
                          float health, int arrows, int bowChargeTicks) {
        this.x = x; this.y = y; this.z = z;
        this.yaw = yaw; this.pitch = pitch;
        this.health = health; this.arrows = arrows; this.bowChargeTicks = bowChargeTicks;
    }

    public static PlayerSnapshot from(ClientPlayerEntity p) {
        int arrows = countArrows(p);
        int charge = 0;
        if (p.isUsingItem() && p.getActiveItem().getItem() == Items.BOW) {
            charge = p.getItemUseTime();
        }
        // 1.16.5: Entity.yaw / Entity.pitch are public float fields.
        return new PlayerSnapshot(p.getX(), p.getY(), p.getZ(), p.yaw, p.pitch,
                p.getHealth(), arrows, charge);
    }

    private static int countArrows(ClientPlayerEntity p) {
        int n = 0;
        // PlayerInventory.main (hotbar+main) and offHand in 1.16.5.
        for (int i = 0; i < p.inventory.main.size(); i++) {
            if (p.inventory.main.get(i).getItem() == Items.ARROW) {
                n += p.inventory.main.get(i).getCount();
            }
        }
        for (int i = 0; i < p.inventory.offHand.size(); i++) {
            if (p.inventory.offHand.get(i).getItem() == Items.ARROW) {
                n += p.inventory.offHand.get(i).getCount();
            }
        }
        return n;
    }

    public void writeFields(StringBuilder sb) {
        sb.append("\"player_xyz\":").append(JsonUtil.vec3(x, y, z)).append(',');
        sb.append("\"player_yaw\":").append(JsonUtil.num(yaw)).append(',');
        sb.append("\"player_pitch\":").append(JsonUtil.num(pitch)).append(',');
        sb.append("\"health\":").append(JsonUtil.num(health)).append(',');
        sb.append("\"arrows\":").append(arrows).append(',');
        sb.append("\"bow_charge_ticks\":").append(bowChargeTicks);
    }
}
