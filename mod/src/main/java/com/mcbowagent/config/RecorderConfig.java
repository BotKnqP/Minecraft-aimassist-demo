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
    public int runtimeCaptureInterval = 4;           // capture/send every N client ticks (raise if laggy).
                                                     // ~4 (5 Hz) matches CPU YOLO throughput so frames don't
                                                     // pile up (less game lag + less latency -> less overshoot).

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
