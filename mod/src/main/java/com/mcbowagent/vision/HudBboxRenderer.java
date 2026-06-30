package com.mcbowagent.vision;

import java.util.List;
import java.util.Locale;

import com.mcbowagent.state.MobSnapshot;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawableHelper;
import net.minecraft.client.util.math.MatrixStack;

/** Draws projected mob bboxes + a status line, for eyeball validation (F9). */
public final class HudBboxRenderer {

    private static final int GREEN = 0xFF00FF00;
    private static final int RED = 0xFFFF5050;
    private static final int YELLOW = 0xFFFFFF00;
    private static final int WHITE = 0xFFFFFFFF;

    private HudBboxRenderer() {}

    /** Draw the live YOLO detection boxes (ESP overlay) the Python client sent for the runtime.
     *  boxes are in the SCALED-GUI coords of the FRAME Python detected; captureYaw/Pitch is the player
     *  view that frame was captured in. Each render we compute the delta against the CURRENT view and
     *  pixel-shift the boxes to compensate — this kills the visible "box trailing during big turns"
     *  caused by the 50-150 ms capture-to-render lag (one mod tick + detection round-trip). */
    public static void renderRuntime(MatrixStack m, MinecraftClient mc, int[][] boxes,
                                     float captureYaw, float capturePitch,
                                     float currentYaw, float currentPitch,
                                     boolean haveCaptureView) {
        int n = boxes == null ? 0 : boxes.length;
        mc.textRenderer.drawWithShadow(m, "[mcbowagent] RUNTIME (vision)  dets:" + n, 4, 4, GREEN);
        if (boxes == null) return;
        int dxPx = 0, dyPx = 0;
        if (haveCaptureView) {
            // focal_px from the CURRENT render FOV (whatever the player has set: 70 / 93 / 110...).
            // Hard-coding 70 used to over-estimate the shift at fov 93 by ~33%, which made boxes visibly
            // overshoot during pans.
            double fov = mc.gameRenderer.getFov(mc.gameRenderer.getCamera(), mc.getTickDelta(), true);
            int h = mc.getWindow().getScaledHeight();
            double focal = (h / 2.0) / Math.tan(Math.toRadians(fov) / 2.0);
            // shift sign: yaw RIGHT (currentYaw > captureYaw) means the world appears to move LEFT in
            // screen coords -> shift the box by a NEGATIVE pixel delta.
            double dy = wrapDeg180(currentYaw - captureYaw);
            double dp = currentPitch - capturePitch;
            dxPx = (int) Math.round(-Math.tan(Math.toRadians(dy)) * focal);
            dyPx = (int) Math.round(-Math.tan(Math.toRadians(dp)) * focal);
        }
        for (int[] b : boxes) {
            if (b.length < 5) continue;
            int color = b[4] == 1 ? RED : (b[4] == 2 ? YELLOW : GREEN);
            int x0 = b[0] + dxPx, y0 = b[1] + dyPx, x1 = b[2] + dxPx, y1 = b[3] + dyPx;
            // outline 2px OUTSIDE the bbox; never feeds back into detection (capture is render-pre-overlay).
            drawRect(m, x0 - 2, y0 - 2, x1 + 2, y1 + 2, color);
        }
    }

    private static float wrapDeg180(float d) {
        d = d % 360.0f;
        if (d >= 180.0f) d -= 360.0f;
        if (d < -180.0f) d += 360.0f;
        return d;
    }

    public static void render(MatrixStack m, MinecraftClient mc, List<MobSnapshot> mobs,
                              boolean recording, boolean oracle, long tick) {
        // status line
        String status = String.format(Locale.US,
                "[mcbowagent] REC:%s  ORACLE:%s  tick:%d  mobs:%d",
                recording ? "ON" : "off", oracle ? "ON" : "off", tick, mobs.size());
        mc.textRenderer.drawWithShadow(m, status, 4, 4, recording ? GREEN : WHITE);

        for (MobSnapshot mob : mobs) {
            if (mob.bbox == null) continue;
            int x0 = mob.bbox[0], y0 = mob.bbox[1], x1 = mob.bbox[2], y1 = mob.bbox[3];
            int color = mob.visible ? GREEN : RED;
            drawRect(m, x0, y0, x1, y1, color);
            String label = String.format(Locale.US, "%s d=%.1f", mob.type, mob.distance);
            mc.textRenderer.drawWithShadow(m, label, x0 + 1, Math.max(0, y0 - 9), color);
        }
    }

    /** 1px rectangle outline using four filled edges. */
    private static void drawRect(MatrixStack m, int x0, int y0, int x1, int y1, int color) {
        DrawableHelper.fill(m, x0, y0, x1, y0 + 1, color);       // top
        DrawableHelper.fill(m, x0, y1 - 1, x1, y1, color);       // bottom
        DrawableHelper.fill(m, x0, y0, x0 + 1, y1, color);       // left
        DrawableHelper.fill(m, x1 - 1, y0, x1, y1, color);       // right
    }
}
