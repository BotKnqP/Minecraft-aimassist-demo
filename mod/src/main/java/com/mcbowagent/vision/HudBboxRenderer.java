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
     *  boxes: [x0,y0,x1,y1,role] in scaled-GUI coords; role 0=other(green) 1=target(red) 2=approach(yellow). */
    public static void renderRuntime(MatrixStack m, MinecraftClient mc, int[][] boxes) {
        int n = boxes == null ? 0 : boxes.length;
        mc.textRenderer.drawWithShadow(m, "[mcbowagent] RUNTIME (vision)  dets:" + n, 4, 4, GREEN);
        if (boxes == null) return;
        for (int[] b : boxes) {
            if (b.length < 5) continue;
            int color = b[4] == 1 ? RED : (b[4] == 2 ? YELLOW : GREEN);
            // 2px outside the bbox just so the outline doesn't hide the silhouette edge. (This is drawn AFTER
            // the detector's frame was captured in renderCapture(), so it never feeds back into detection.)
            drawRect(m, b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2, color);
        }
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
