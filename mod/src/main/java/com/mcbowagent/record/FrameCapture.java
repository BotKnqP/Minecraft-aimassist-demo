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

    /** Grab the current frame, downscale to (targetW,targetH), return PNG bytes
     *  (in-memory) for the runtime socket bridge. Null on error.
     *  TODO(verify): NativeImage.getBytes() is the 1.16.5 Yarn name returning the
     *  PNG-encoded bytes (sibling of writeFile); confirm on first IDE compile. */
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
}
