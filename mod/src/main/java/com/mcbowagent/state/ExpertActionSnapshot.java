package com.mcbowagent.state;

import com.mcbowagent.config.RecorderConfig;
import com.mcbowagent.record.JsonUtil;

import net.minecraft.client.MinecraftClient;
import net.minecraft.util.math.MathHelper;

/**
 * The expert action ACTUALLY taken this tick. camera is a per-tick RELATIVE
 * delta in degrees (NOT absolute), derived from how the view actually moved 鈥? * this unifies human play and the oracle (which drives the view the same way).
 */
public final class ExpertActionSnapshot {

    public final double dPitch, dYaw;
    public final int forward, back, left, right, jump, sprint, sneak, use, hotbar;
    public final int targetEntityId;

    private ExpertActionSnapshot(double dPitch, double dYaw, int forward, int back,
                                 int left, int right, int jump, int sprint, int sneak,
                                 int use, int hotbar, int targetEntityId) {
        this.dPitch = dPitch; this.dYaw = dYaw;
        this.forward = forward; this.back = back; this.left = left; this.right = right;
        this.jump = jump; this.sprint = sprint; this.sneak = sneak; this.use = use;
        this.hotbar = hotbar; this.targetEntityId = targetEntityId;
    }

    /**
     * @param prevYaw,prevPitch the player's look angles on the PREVIOUS tick
     *        (so we can recover the relative camera delta actually applied).
     */
    public static ExpertActionSnapshot capture(MinecraftClient mc, float curYaw, float curPitch,
                                               float prevYaw, float prevPitch, int targetEntityId) {
        double dYaw = MathHelper.wrapDegrees(curYaw - prevYaw);
        double dPitch = curPitch - prevPitch;
        // clip to +-10 deg/tick to match the VPT camera range (residual carries to next tick).
        dYaw = clip(dYaw, RecorderConfig.CAMERA_MAX_STEP_DEG);
        dPitch = clip(dPitch, RecorderConfig.CAMERA_MAX_STEP_DEG);

        int sel = mc.player.inventory.selectedSlot + 1;   // 1..9
        return new ExpertActionSnapshot(
                dPitch, dYaw,
                pressed(mc.options.keyForward), pressed(mc.options.keyBack),
                pressed(mc.options.keyLeft), pressed(mc.options.keyRight),
                pressed(mc.options.keyJump), pressed(mc.options.keySprint),
                pressed(mc.options.keySneak), pressed(mc.options.keyUse),
                sel, targetEntityId);
    }

    private static int pressed(net.minecraft.client.option.KeyBinding k) {
        return k.isPressed() ? 1 : 0;
    }

    private static double clip(double v, double m) {
        return v < -m ? -m : (v > m ? m : v);
    }

    /** JSON object (nested under "action"). Order matches data_schema.ExpertAction. */
    public void writeObject(StringBuilder sb) {
        sb.append('{');
        sb.append("\"camera\":[").append(JsonUtil.num(dPitch)).append(',')
          .append(JsonUtil.num(dYaw)).append("],");
        sb.append("\"forward\":").append(forward).append(',');
        sb.append("\"back\":").append(back).append(',');
        sb.append("\"left\":").append(left).append(',');
        sb.append("\"right\":").append(right).append(',');
        sb.append("\"jump\":").append(jump).append(',');
        sb.append("\"sprint\":").append(sprint).append(',');
        sb.append("\"sneak\":").append(sneak).append(',');
        sb.append("\"use\":").append(use).append(',');
        sb.append("\"hotbar\":").append(hotbar).append(',');
        sb.append("\"target_entity_id\":").append(targetEntityId);
        sb.append('}');
    }
}

