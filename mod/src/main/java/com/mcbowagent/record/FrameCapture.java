package com.mcbowagent.record;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import org.lwjgl.system.MemoryUtil;

import com.mojang.blaze3d.systems.RenderSystem;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gl.Framebuffer;
import net.minecraft.client.texture.NativeImage;

/**
 * Captures the main framebuffer and writes it DOWNSCALED to the same resolution
 * the bboxes were projected in (getScaledWidth/Height), so the saved frame and
 * the YOLO labels share one coordinate space (dataset.py reads the image's W,H).
 *
 * Must run on the client/render thread (END_CLIENT_TICK is) 鈥?it issues GL calls.
 * The vanilla HUD (hotbar/crosshair) is in the frame, which is consistent with
 * what the agent sees at runtime; OUR debug bbox overlay is suppressed while
 * recording (see McBowAgentMod) so it never contaminates training frames.
 *
 * TODO(verify on first IDE compile): NativeImage.loadFromTextureImage /
 * resizeSubRectTo / writeTo and Framebuffer.textureWidth/getColorAttachment are
 * the 1.16.5 Yarn names used by vanilla's screenshot code; confirm if any differ.
 */
public final class FrameCapture {

    public String frameRelPath(long tick) {
        return String.format(Locale.US, "frames/frame_%06d.png", tick);
    }

    /** Render-thread only: GL readback + vertical mirror + downscale to (targetW, targetH). Returns the
     *  small NativeImage OWNED BY THE CALLER (must be close()d after the consumer is done with it). The
     *  PNG encode + disk write are NOT done here — hand the result to AsyncFrameWriter. Null if the
     *  framebuffer isn't ready yet. */
    public NativeImage captureSmallImage(MinecraftClient mc, int targetW, int targetH) {
        Framebuffer fb = mc.getFramebuffer();
        int w = fb.textureWidth;
        int h = fb.textureHeight;
        if (w <= 0 || h <= 0) return null;
        NativeImage full = new NativeImage(w, h, false);
        try {
            RenderSystem.bindTexture(fb.getColorAttachment());
            full.loadFromTextureImage(0, true);
            full.mirrorVertically();
            if (targetW > 0 && targetH > 0 && (targetW != w || targetH != h)) {
                NativeImage small = new NativeImage(targetW, targetH, false);
                full.resizeSubRectTo(0, 0, w, h, small);
                return small;
            }
            // edge case: target == source. Return a copy so the caller can close it independently.
            NativeImage copy = new NativeImage(w, h, false);
            full.resizeSubRectTo(0, 0, w, h, copy);
            return copy;
        } finally {
            full.close();
        }
    }

    /** Grab the current frame, downscale to (targetW,targetH), write a PNG. Non-fatal on error. */
    public void capture(MinecraftClient mc, Path outFile, int targetW, int targetH) {
        Framebuffer fb = mc.getFramebuffer();
        int w = fb.textureWidth;
        int h = fb.textureHeight;
        if (w <= 0 || h <= 0) return;

        NativeImage full = new NativeImage(w, h, false);
        try {
            RenderSystem.bindTexture(fb.getColorAttachment());
            full.loadFromTextureImage(0, true);   // read the bound GL color texture
            full.mirrorVertically();              // GL framebuffers are bottom-up

            NativeImage outImg = full;
            boolean resized = false;
            if (targetW > 0 && targetH > 0 && (targetW != w || targetH != h)) {
                NativeImage small = new NativeImage(targetW, targetH, false);
                full.resizeSubRectTo(0, 0, w, h, small);   // downscale full -> small
                outImg = small;
                resized = true;
            }
            try {
                if (outFile.getParent() != null) {
                    Files.createDirectories(outFile.getParent());
                }
                outImg.writeFile(outFile.toFile());          // PNG encode + write
            } finally {
                if (resized) {
                    outImg.close();
                }
            }
        } catch (IOException | RuntimeException e) {
            System.err.println("[mcbowagent] frame capture failed: " + e.getMessage());
        } finally {
            full.close();
        }
    }

    /** Grab the current frame, downscale to (targetW,targetH), return PNG bytes (in-memory) for the
     *  runtime socket bridge. Null on error. PNG encode is the slow path (~10-20 ms on the render thread);
     *  see captureRawBytes for the fast path that emits raw BGR pixels instead. */
    public byte[] captureBytes(MinecraftClient mc, int targetW, int targetH) throws IOException {
        Framebuffer fb = mc.getFramebuffer();
        int w = fb.textureWidth;
        int h = fb.textureHeight;
        if (w <= 0 || h <= 0) return null;          // not ready yet -> soft skip (caller logs)

        NativeImage full = new NativeImage(w, h, false);
        try {
            RenderSystem.bindTexture(fb.getColorAttachment());
            full.loadFromTextureImage(0, true);
            full.mirrorVertically();

            NativeImage outImg = full;
            boolean resized = false;
            if (targetW > 0 && targetH > 0 && (targetW != w || targetH != h)) {
                NativeImage small = new NativeImage(targetW, targetH, false);
                full.resizeSubRectTo(0, 0, w, h, small);
                outImg = small;
                resized = true;
            }
            try {
                return outImg.getBytes();   // PNG-encoded bytes; throws on encode error
            } finally {
                if (resized) {
                    outImg.close();
                }
            }
        } finally {
            full.close();   // RuntimeException / IOException propagate to the caller (RuntimeBridge logs)
        }
    }

    /**
     * Fast path for the live runtime: capture + downscale + emit RAW BGR pixels.
     *
     * Header (magic 'W' — current versioned format, 15 bytes):
     *   [magic='W'][W:u16 BE][H:u16 BE][capture_unix_ms:u64 BE][fov_x100:u16 BE]
     * Payload: 15 + W*H*3 BGR bytes.
     *
     * Why fov_x100: the user's runtime FOV is whatever they set in Video Settings (often 93, sometimes
     * 70, plus mods). Python's bearing / range / focal math must use the SAME FOV that the projection
     * matrix used for THIS frame, or distances and bearings drift. fov_x100 = (int) round(fov * 100),
     * range 1..36000 fits in u16. Pass the SAME FOV value that the world was rendered with this frame:
     *   double fov = mc.gameRenderer.getFov(mc.gameRenderer.getCamera(), tickDelta, true);
     *
     * Python's protocol.decode_frame sniffs the first byte and routes
     *   'W' (0x57) -> raw-v2 + capture_ms + fov,  meta = {"capture_ms", "fov_deg"}
     *   'V' (0x56) -> raw-v1 + capture_ms (legacy), meta = {"capture_ms"}
     *   'R' (0x52) -> raw (legacy, no meta)
     *   0x89       -> PNG (legacy slow path)
     */
    public byte[] captureRawBytes(MinecraftClient mc, int targetW, int targetH, double fovDeg) {
        Framebuffer fb = mc.getFramebuffer();
        int w = fb.textureWidth;
        int h = fb.textureHeight;
        if (w <= 0 || h <= 0) return null;

        NativeImage full = new NativeImage(w, h, false);
        try {
            RenderSystem.bindTexture(fb.getColorAttachment());
            full.loadFromTextureImage(0, true);
            full.mirrorVertically();

            NativeImage outImg = full;
            boolean resized = false;
            int ow = w, oh = h;
            if (targetW > 0 && targetH > 0 && (targetW != w || targetH != h)) {
                NativeImage small = new NativeImage(targetW, targetH, false);
                full.resizeSubRectTo(0, 0, w, h, small);
                outImg = small;
                resized = true;
                ow = targetW;
                oh = targetH;
            }
            try {
                // Versioned raw header 'W': capture_ms + fov so Python uses the SAME fov the projection
                // matrix used for this frame (user runs 93°, not the hard-coded 70°).
                long captureMs = System.currentTimeMillis();
                int fovX100 = Math.max(0, Math.min(36000, (int) Math.round(fovDeg * 100.0)));
                byte[] out = new byte[15 + ow * oh * 3];
                out[0] = (byte) 'W';
                out[1] = (byte) ((ow >>> 8) & 0xff);
                out[2] = (byte) (ow & 0xff);
                out[3] = (byte) ((oh >>> 8) & 0xff);
                out[4] = (byte) (oh & 0xff);
                for (int i = 0; i < 8; i++) {
                    out[5 + i] = (byte) ((captureMs >>> (8 * (7 - i))) & 0xff);
                }
                out[13] = (byte) ((fovX100 >>> 8) & 0xff);
                out[14] = (byte) (fovX100 & 0xff);
                // Bulk byte-access: NativeImage stores pixels in memory as B,G,R,A bytes (verified by the
                // old getPixelColor decoder: c & 0xff = B). We pull all 4-byte pixels in one bulk get(),
                // then a tight Java loop strips the A byte. At 427×240 this is ~0.4 ms vs ~6 ms for the
                // 102 k JNI getPixelColor calls — the difference between sustainable 60 Hz and choking.
                final int npx = ow * oh;
                byte[] rgba = new byte[npx * 4];
                ByteBuffer src = MemoryUtil.memByteBuffer(outImg.pointer, npx * 4);
                src.position(0).limit(npx * 4);
                src.get(rgba);
                int o = 15;
                for (int i = 0, s = 0; i < npx; i++, s += 4) {
                    out[o++] = rgba[s];                          // B (byte 0 of the BGRA pixel)
                    out[o++] = rgba[s + 1];                      // G
                    out[o++] = rgba[s + 2];                      // R
                }
                return out;
            } finally {
                if (resized) {
                    outImg.close();
                }
            }
        } finally {
            full.close();
        }
    }
}
