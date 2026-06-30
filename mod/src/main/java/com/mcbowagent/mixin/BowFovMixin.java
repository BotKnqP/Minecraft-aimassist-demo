package com.mcbowagent.mixin;

import com.mcbowagent.McBowAgentMod;

import net.minecraft.client.network.AbstractClientPlayerEntity;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.Constant;
import org.spongepowered.asm.mixin.injection.ModifyConstant;

/**
 * Suppress the bow-charge FOV zoom while the vision runtime owns the view (F7 on).
 *
 * WHY THIS MIXIN EXISTS
 * ---------------------
 * The Python vision pipeline assumes a CONSTANT 70 degree vertical FOV:
 *   focal_px = H / (2 * tan(35 deg))
 * This focal length feeds bearing_from_bbox, distance = k / bbox_h, the render-time
 * ESP-box drift compensation, and the < 2 deg bow-align fire gate. Vanilla 1.16.5
 * progressively shrinks the FOV by up to 15% while the player charges a bow (snap
 * back on release). If we let that fire mid-charge:
 *   - same screen pixel = a smaller world angle  -> Python over-estimates d_yaw
 *   - same bbox height  = a closer target        -> range under-estimates -> bad drop comp
 *   - ESP boxes drift in pixels (stale focal)    -> the "trailing boxes" bug returns
 *   - the 2 deg align gate becomes ~2.4 deg in unzoomed coords -> looser fire
 * So while the runtime owns the view we need a stable FOV.
 *
 * WHY HERE AND NOT IN GameRenderer.getFov
 * ---------------------------------------
 * In yarn 1.16.5 the bow zoom is NOT applied inside GameRenderer.getFov. It is
 * applied one level up, in AbstractClientPlayerEntity.getSpeed(), whose return
 * value ClientPlayerEntity.tick() writes into GameRenderer.movementFovMultiplier.
 * getFov then lerps last/current movementFovMultiplier (gated by options.fovEffectScale)
 * and applies the death-cam squeeze + underwater 60/70 factor. Touching getFov would
 * either disturb every FOV contributor (sprint + slowness + bow + potions) or force
 * us to reconstruct vanilla math to back out just the bow piece. getSpeed() is the
 * one place where the bow contribution exists as a separable multiplicative factor.
 *
 * WHY @ModifyConstant ON 0.15F
 * ----------------------------
 * Vanilla bow branch (semantically):
 *     f *= (1.0F - p * 0.15F);   // p = clamped, squared charge progress
 * The literal 0.15F appears EXACTLY ONCE in AbstractClientPlayerEntity and only inside
 * the `if (item == Items.BOW)` arm. Returning 0.0F from this modifier rewrites the line
 * to `f *= (1 - p * 0) = f * 1` -- an exact, bit-identical no-op for the bow case.
 * Zero recomputation, zero drift, no collateral on sprint / slowness / potion /
 * fovEffectScale / death-cam / underwater FOV.
 * require = 1, allow = 1 makes the build fail loudly if a future mappings/refactor
 * bump moves the literal -- better than silently regressing to a zooming pipeline.
 *
 * GATING
 * ------
 * Mirrors MouseLookMixin: read the static volatile McBowAgentMod.RUNTIME_OWNS_VIEW
 * (set in onEndTick when F7 toggles config.runtimeActive). When the agent is not
 * running, vanilla bow zoom is fully preserved.
 *
 * SCOPE
 * -----
 * Targets AbstractClientPlayerEntity rather than ClientPlayerEntity because
 * getSpeed() is defined on the abstract base. Other players' getSpeed() is irrelevant
 * to local-camera FOV, which is driven solely by the local ClientPlayerEntity, so
 * binding to the base class is safe and matches how Mojang authored the method.
 */
@Mixin(AbstractClientPlayerEntity.class)
public class BowFovMixin {

    @ModifyConstant(
            method = "getSpeed",
            constant = @Constant(floatValue = 0.15F),
            require = 1,
            allow = 1)
    private float mcbow$suppressBowFov(float original) {
        return McBowAgentMod.RUNTIME_OWNS_VIEW ? 0.0F : original;
    }
}
