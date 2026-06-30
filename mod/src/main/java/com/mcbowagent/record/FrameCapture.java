package com.mcbowagent.record;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

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
     * Fast path for the live runtime: same capture+downscale as captureBytes, but skip PNG encoding and
     * emit RAW BGR pixels prefixed by a 13-byte header
     *   [magic='V'][W:u16 BE][H:u16 BE][capture_unix_ms:u64 BE]
     * Payload layout: 13 + W*H*3 bytes.
     *
     * The capture_unix_ms timestamp (System.currentTimeMillis at the moment of GL readback) lets the Python
     * tracker subtract "mod turn since capture" from the measured bearing, killing the overshoot caused by
     * staleness (~50-150 ms detection latency on CPU YOLO).
     *
     * Python's protocol.decode_frame sniffs the first byte and routes 'V' raw-v2 / 'R' raw-v1 / 0x89 PNG.
     */
    public byte[] captureRawBytes(MinecraftClient mc, int targetW, int targetH) {
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
                // Versioned raw header 'V': adds an 8-byte capture timestamp so the Python tracker can
                // subtract "mod turn since capture" from the measured bearing.
                long captureMs = System.currentTimeMillis();
                byte[] out = new byte[13 + ow * oh * 3];
                out[0] = (byte) 'V';
                out[1] = (byte) ((ow >>> 8) & 0xff);
                out[2] = (byte) (ow & 0xff);
                out[3] = (byte) ((oh >>> 8) & 0xff);
                out[4] = (byte) (oh & 0xff);
                for (int i = 0; i < 8; i++) {
                    out[5 + i] = (byte) ((captureMs >>> (8 * (7 - i))) & 0xff);
                }
                int o = 13;
                for (int y = 0; y < oh; y++) {
                    for (int x = 0; x < ow; x++) {
                        int c = outImg.getPixelColor(x, y);   // ABGR: A=high, then B, G, R
                        out[o++] = (byte) (c & 0xff);           // B
                        out[o++] = (byte) ((c >>> 8) & 0xff);   // G
                        out[o++] = (byte) ((c >>> 16) & 0xff);  // R
                    }
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
