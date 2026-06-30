package com.mcbowagent.config;

/**
 * Runtime toggles + constants for the recorder. Physics constants here MUST stay
 * in sync with python/mc_bow_agent/constants.py — the M0 parity gate checks both.
 */
public final class RecorderConfig {

    // --- runtime toggles (flipped by F8/F9/F10/F7) ---
    public boolean recording = false;
    public boolean hudBboxes = true;
    public boolean oracleEnabled = false;
    public boolean runtimeActive = false;            // F7: vision->bow socket bridge

    // --- runtime bridge ---
    public int runtimePort = 5555;                   // localhost TCP port for the Python client
    public boolean runtimeRawFrame = true;           // send raw BGR pixels instead of PNG (skips the slow
                                                     // PNG encode on the render thread; Python sniffs 'R' vs
                                                     // PNG magic so legacy peers still work). Toggle false to
                                                     // force PNG if a peer can't handle raw.
    public int recordCaptureInterval = 2;            // RECORD a frame every N client ticks (~10 Hz default).
                                                     // The recorder is the heavy path: full-res GL readback
                                                     // + downscale + PNG encode + disk write per saved frame.
                                                     // 10 Hz training data is plenty (the existing dataset is
                                                     // ~10 Hz) and at 20 Hz the render thread chokes.
    public int runtimeCaptureHz = 60;                // target capture rate (Hz) when the runtime is on. Driven
                                                     // by the RENDER thread's own clock, not by client tick —
                                                     // so it can go above the 20 Hz tick rate. Backpressure
                                                     // (frameToSend != null) prevents pile-up when Python is
                                                     // slower than this. Lower to 30/20/10 if your GPU can't
                                                     // sustain the readback cost (~0.5-2 ms per frame).
    public int runtimeCaptureInterval = 2;           // DEPRECATED legacy: kept as a fallback for the recorder
                                                     // path (F8); F7 runtime now uses runtimeCaptureHz above.

    // --- scan / output ---
    public double scanRadius = 40.0;                 // blocks; only log hostiles within this
    public String outputBaseDir = "D:\\projects\\mc-bow-agent\\runs";

    // --- camera / action (mirror python constants) ---
    public static final double CAMERA_MAX_STEP_DEG = 10.0;   // <= +-10 deg/tick (VPT clip)

    // --- arrow physics (mirror python constants.py) ---
    public static final double ARROW_DRAG = 0.99;
    public static final double ARROW_GRAVITY = 0.05;
    public static final double BOW_FULL_CHARGE_SPEED = 3.0;  // blocks/tick at full draw
    public static final int FULL_CHARGE_TICKS = 20;
    public static final int MIN_FIRE_TICKS = 3;

    // --- mob types of interest for v0 ---
    public static final String[] V0_MOB_TYPES = {"zombie", "skeleton", "creeper", "spider"};
}
