package com.mcbowagent.mixin;

import com.mcbowagent.McBowAgentMod;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * While the vision runtime owns the view (F7 on), suppress the PHYSICAL mouse's effect on the player's look.
 *
 * The bot drives the camera by writing yaw/pitch directly each tick; Minecraft ALSO applies the mouse's
 * cursor delta to the same yaw/pitch via {@code Entity.changeLookDirection} (called by {@code Mouse}). The two
 * fight over the view ("抢鼠标") — the mouse drags it to the middle/ground and the bot keeps loosing arrows at
 * empty space. Cancelling the mouse-look path (only for the client player, only while the runtime owns the view)
 * gives the bot exclusive control. Movement keys (WASD) are unaffected; F7-off restores normal mouse look.
 */
@Mixin(Entity.class)
public class MouseLookMixin {

    @Inject(method = "changeLookDirection", at = @At("HEAD"), cancellable = true, require = 0)
    private void mcbow$suppressMouseLook(double cursorDeltaX, double cursorDeltaY, CallbackInfo ci) {
        if (McBowAgentMod.RUNTIME_OWNS_VIEW && (Object) this instanceof ClientPlayerEntity) {
            ci.cancel();
        }
    }
}
