package com.mcbowagent.oracle;

import com.mcbowagent.config.RecorderConfig;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.item.Items;
import net.minecraft.util.math.MathHelper;

/**
 * Drives the view + bow ONLY by rotating the camera and toggling the use key.
 * NEVER teleports the player or edits any entity position — this is a legal
 * aimbot (rotation + input), nothing more.
 *
 * Bow control is the SIMPLE pre-overlay logic: draw while turning, release (fire)
 * once aligned and fully charged. (The near-gate / post-fire-cooldown / hysteresis /
 * finite-guard experiments were all reverted — they didn't help.)
 */
public final class BowMacroController {

    private static final double ALIGN_DEG = 2.0;
    private static final double TURN_GAIN = 0.45;        // move 45% of the residual per tick (damps overshoot)
    private static final double LIVE_MAX_STEP_DEG = 8.0; // per-tick clamp for the live aim (<= the 10deg VPT cap)
    private static final double MOVE_DEADZONE_DEG = 1.2; // within this, command ZERO turn (kills sub-deg jitter)

    /** Step the player's look toward (targetYaw,targetPitch): proportional gain + deadzone, <= 8 deg/tick.
     *  Returns true when essentially on target (FULL error within ALIGN_DEG, so fire accuracy is unchanged). */
    public boolean stepView(ClientPlayerEntity p, double targetYaw, double targetPitch) {
        double dYaw = MathHelper.wrapDegrees(targetYaw - p.yaw);
        double dPitch = targetPitch - p.pitch;
        double sYaw = Math.abs(dYaw) < MOVE_DEADZONE_DEG ? 0.0 : clip(dYaw * TURN_GAIN, LIVE_MAX_STEP_DEG);
        double sPitch = Math.abs(dPitch) < MOVE_DEADZONE_DEG ? 0.0 : clip(dPitch * TURN_GAIN, LIVE_MAX_STEP_DEG);
        p.yaw += (float) sYaw;
        p.pitch = (float) MathHelper.clamp(p.pitch + (float) sPitch, -90.0f, 90.0f);
        return Math.abs(dYaw) < ALIGN_DEG && Math.abs(dPitch) < ALIGN_DEG;
    }

    /** Hold the bow to charge; release (fire) once aligned and fully drawn. */
    public void manageBow(MinecraftClient mc, boolean aligned) {
        ClientPlayerEntity p = mc.player;
        boolean hasBow = p.getMainHandStack().getItem() == Items.BOW;
        if (!hasBow) {
            setUse(mc, false);
            return;
        }
        int charge = (p.isUsingItem() && p.getActiveItem().getItem() == Items.BOW)
                ? p.getItemUseTime() : 0;
        if (aligned && charge >= RecorderConfig.FULL_CHARGE_TICKS) {
            setUse(mc, false);   // release -> looses the arrow this tick
        } else {
            setUse(mc, true);    // keep drawing (also while still turning)
        }
    }

    public void releaseBow(MinecraftClient mc) {
        setUse(mc, false);
    }

    /** Stop drawing the bow WITHOUT loosing an arrow — the safe neutral when there is no valid/fresh action.
     *  clearActiveItem cancels the draw without the release (fire) effect. */
    public void stopUsing(MinecraftClient mc) {
        setUse(mc, false);
        if (mc.player != null) {
            mc.player.clearActiveItem();
        }
    }

    private void setUse(MinecraftClient mc, boolean pressed) {
        mc.options.keyUse.setPressed(pressed);
    }

    private static double clip(double v, double m) {
        return v < -m ? -m : (v > m ? m : v);
    }
}
