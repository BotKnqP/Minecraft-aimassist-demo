package com.mcbowagent.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mcbowagent.config.RecorderConfig;
import com.mcbowagent.oracle.BowMacroController;
import com.mcbowagent.record.FrameCapture;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.text.LiteralText;

/**
 * v1 runtime bridge (vision -> bow), hardened.
 *
 * DESIGN: the GL frame capture + the bow control run on the CLIENT/RENDER thread
 * (controlTick), but ALL socket I/O runs on a separate NET thread, so a slow or
 * absent Python client can never freeze the game. The net thread fills
 * {@code latestAction} + {@code lastActionTimeMs}; controlTick reads them and acts
 * only on a FRESH action (<= {@value #STALE_MS} ms). With no client, or a stale /
 * missing action, the control does nothing and the bow is stopped WITHOUT firing.
 *
 * Wire protocol (docs/RUNTIME_PROTOCOL.md): 4-byte big-endian length + payload.
 * mod -> py = PNG frame (scaled-GUI res). py -> mod = action JSON
 * {has_target, d_yaw, d_pitch, range, fire_ok, n_det}.
 */
public final class RuntimeBridge {

    private static final Logger LOGGER = LogManager.getLogger("mcbowagent");
    private static final long STALE_MS = 200;          // action older than this -> neutral
    private static final long GRACE_MS = 500;          // keep the bow drawn this long after losing a target
    private static final int READ_TIMEOUT_MS = 0;      // 0 = infinite: do NOT auto-drop the client on
                                                       // a slow reply (cancelled per request). A closed/
                                                       // reset socket is still detected (read throws);
                                                       // a truly hung-but-open client is cleared by F7.
    private static final long VERBOSE_TICKS = 30;      // log first N at INFO, then DEBUG (avoid flood)
    private static final int CAPTURE_FAIL_LIMIT = 20;  // consecutive capture exceptions before dropping the
                                                       // client (a transient GL/encode hiccup must NOT churn
                                                       // the connection -> skip the frame and keep going)

    private final RecorderConfig cfg;
    private final FrameCapture frameCapture = new FrameCapture();
    private final BowMacroController controller = new BowMacroController();

    private volatile ServerSocket server;
    private volatile Socket client;
    private Thread netThread;

    // main thread -> net thread: latest captured frame (latest-wins)
    private final Object frameLock = new Object();
    private byte[] frameToSend = null;

    // net thread -> main thread
    private volatile RuntimeAction latestAction = null;
    private volatile long lastActionTimeMs = 0;
    private volatile int[][] latestBoxes = null;   // [x0,y0,x1,y1,role] per detection, for the HUD ESP overlay

    /** Latest detection boxes (scaled-GUI coords) for the overlay, or null if absent/STALE. Render thread reads
     *  it. The freshness gate (mirrors the action's STALE handling, looser for the eye) stops frozen boxes from
     *  lingering on screen when the Python client stalls. */
    public int[][] getLatestBoxes() {
        return (latestBoxes != null && System.currentTimeMillis() - lastActionTimeMs <= 1000L) ? latestBoxes : null;
    }

    private long frameSeq = 0;               // net thread only
    private volatile long actionSeq = 0;     // net writes, main reads (turn gating)
    private long controlTicks = 0;           // main thread only
    private long lastAppliedActionSeq = -1;  // main thread only (which action we last TURNED on)
    private boolean lastAligned = false;     // main thread only
    private boolean prevAligned = false;     // main thread only (alignment of the PREVIOUS applied action)
    private long lastSentCrc = 0L;           // main thread only — dedup via CRC32 (avoids holding a 300 KB byte[])
    private boolean haveLastSentCrc = false;
    private final java.util.zip.CRC32 dedupCrc = new java.util.zip.CRC32();
    private long lastFireableMs = 0;         // main thread only (last time we had a fireable target)
    private int captureFailStreak = 0;       // main thread only (consecutive capture exceptions)
    private volatile boolean captureRequested = false;  // tick requests; the render thread does the readback
    private float homeYaw = 0f;              // the yaw the player faced at F7 (the arena centre)
    private boolean haveHome = false;        // when idle, ease yaw back to homeYaw -> turn to the other side

    public RuntimeBridge(RecorderConfig cfg) {
        this.cfg = cfg;
    }

    // ---------------- lifecycle (main thread, on F7) ----------------

    /** F7 ON: reset control state, drop the bow, open the server. */
    public synchronized void start(MinecraftClient mc, int port) {
        if (server != null) return;
        resetControl();
        controller.stopUsing(mc);
        if (mc.player != null) { homeYaw = mc.player.yaw; haveHome = true; }   // remember where the arena is
        try {
            server = new ServerSocket(port);
            LOGGER.info("[mcbowagent] runtime bridge: server started on port {}", port);
            netThread = new Thread(this::netLoop, "mcbowagent-net");
            netThread.setDaemon(true);
            netThread.start();
        } catch (IOException e) {
            LOGGER.error("[mcbowagent] runtime bridge: failed to start on port " + port, e);
            server = null;
        }
    }

    /** F7 OFF: stop the server, drop any client, reset control, drop the bow. */
    public synchronized void stop(MinecraftClient mc) {
        ServerSocket s = server;
        server = null;            // signals netLoop to exit
        closeClient();
        resetControl();
        controller.stopUsing(mc);
        if (s != null) {
            try { s.close(); } catch (IOException ignored) {}
        }
        LOGGER.info("[mcbowagent] runtime bridge: stopped");
    }

    private void resetControl() {
        latestAction = null;
        latestBoxes = null;
        lastActionTimeMs = 0;
        lastAppliedActionSeq = -1;
        lastAligned = false;
        prevAligned = false;
        haveLastSentCrc = false;
        lastFireableMs = 0;
        captureFailStreak = 0;
        captureRequested = false;
        synchronized (frameLock) {
            frameToSend = null;
            frameLock.notifyAll();
        }
    }

    // ---------------- net thread: accept + send-frame / recv-action ----------------

    private void netLoop() {
        while (server != null && !server.isClosed()) {
            Socket sock;
            try {
                sock = server.accept();
                sock.setTcpNoDelay(true);
                sock.setSoTimeout(READ_TIMEOUT_MS);
            } catch (IOException e) {
                return;   // server closed -> exit thread
            }
            LOGGER.info("[mcbowagent] runtime bridge: client accepted {}", sock.getRemoteSocketAddress());
            client = sock;
            frameSeq = 0;
            actionSeq = 0;
            try (DataInputStream in = new DataInputStream(sock.getInputStream());
                 DataOutputStream out = new DataOutputStream(sock.getOutputStream())) {
                serveClient(in, out);
            } catch (IOException e) {
                LOGGER.error("[mcbowagent] runtime bridge: socket exception", e);   // req 1/6
            } finally {
                LOGGER.info("[mcbowagent] runtime bridge: client disconnected");
                closeClient();   // req 3: clears latestAction; main thread stopUsing next tick
            }
        }
    }

    private void serveClient(DataInputStream in, DataOutputStream out) throws IOException {
        // PIPELINED: writer (this thread) sends frames as soon as the render thread produces them; a
        // separate reader thread consumes actions as fast as Python returns them. This breaks the old
        // lock-step ceiling: round-trip latency no longer bounds capture rate.  Latest-wins ordering on
        // the mod side is preserved by actionSeq + lastAppliedActionSeq in controlTick — an action is
        // applied at most once even if multiple arrive between control ticks.
        final Thread reader = new Thread(() -> readActionsLoop(in), "mcbowagent-net-reader");
        reader.setDaemon(true);
        reader.start();
        try {
            while (server != null && client != null) {
                byte[] payload = takeFrame();
                if (payload == null) continue;        // no frame ready yet; re-check liveness
                out.writeInt(payload.length);
                out.write(payload);
                out.flush();
                frameSeq++;
                trace(frameSeq <= VERBOSE_TICKS, "[mcbowagent] send frame success: seq={} bytes={}",
                        frameSeq, payload.length);
            }
        } finally {
            // close input -> reader.readInt throws -> reader exits cleanly
            try { in.close(); } catch (IOException ignored) {}
            try { reader.join(2000); } catch (InterruptedException ignored) {}
        }
    }

    /** A parsed action + its arrival timestamp and sequence — published as a single immutable object so
     *  controlTick sees a consistent snapshot. Removes the per-frame JsonObject allocation in the hot path. */
    static final class RuntimeAction {
        final boolean hasTarget;
        final double dYaw;
        final double dPitch;
        final boolean fireOk;
        final long timeMs;
        final long seq;
        RuntimeAction(boolean ht, double dy, double dp, boolean fo, long timeMs, long seq) {
            this.hasTarget = ht; this.dYaw = dy; this.dPitch = dp; this.fireOk = fo;
            this.timeMs = timeMs; this.seq = seq;
        }
    }

    private static final int MAX_ACTION_BYTES = 64 * 1024;   // 19 B header + 10 B per box * 100 max ~ 1 KB;
                                                             // 64 KB is generous and catches corrupt/junk
                                                             // length bytes BEFORE a giant alloc OOMs the JVM.

    /** Reader thread body: consume action frames forever; update latestAction + actionSeq + latestBoxes.
     *  Each payload is sniffed on the FIRST BYTE: 'A' = binary (fast path, no JsonParser/String alloc),
     *  '{' = legacy JSON. Both produce a RuntimeAction + an int[][] of boxes. */
    private void readActionsLoop(DataInputStream in) {
        try {
            while (server != null && client != null) {
                int n = in.readInt();
                if (n < 0 || n > MAX_ACTION_BYTES) {
                    throw new IOException("bad action length: " + n + " (out of [0," + MAX_ACTION_BYTES + "])");
                }
                byte[] buf = new byte[n];
                in.readFully(buf);
                boolean ht; double dy, dp; boolean fo;
                int[][] boxes;
                if (n > 0 && buf[0] == (byte) 'A') {
                    // binary: see protocol.py encode_action_bin for the exact layout
                    ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN);
                    bb.get();                          // magic
                    ht = bb.get() != 0;
                    dy = bb.getFloat();
                    dp = bb.getFloat();
                    bb.getFloat();                     // range (unused by controlTick; still parsed for completeness)
                    fo = bb.get() != 0;
                    bb.getShort();                     // n_det (info-only)
                    int nBoxes = bb.getShort() & 0xffff;
                    boxes = new int[nBoxes][];
                    for (int i = 0; i < nBoxes; i++) {
                        int x0 = bb.getShort();
                        int y0 = bb.getShort();
                        int x1 = bb.getShort();
                        int y1 = bb.getShort();
                        int role = bb.getShort();
                        boxes[i] = new int[]{x0, y0, x1, y1, role};
                    }
                } else {
                    // legacy JSON path (kept for tests / older Python clients)
                    JsonObject o = new JsonParser()       // Gson 2.8.0 (MC 1.16.5)
                            .parse(new String(buf, StandardCharsets.UTF_8)).getAsJsonObject();
                    ht = o.has("has_target") && o.get("has_target").getAsBoolean();
                    dy = optD(o, "d_yaw");
                    dp = optD(o, "d_pitch");
                    fo = optB(o, "fire_ok");
                    boxes = parseBoxes(o);
                }
                // ATOMIC publish: bundle action+time+seq into one immutable, publish via a single volatile
                // store so controlTick can never see a torn (latestAction old / actionSeq new) snapshot.
                long now = System.currentTimeMillis();
                long seq = actionSeq + 1;
                latestBoxes = boxes;                   // boxes published independently — overlay-only, no
                                                       // freshness/seq coupling needed
                latestAction = new RuntimeAction(ht, dy, dp, fo, now, seq);
                lastActionTimeMs = now;                // kept for any code still reading the legacy field
                actionSeq = seq;
                trace(seq <= VERBOSE_TICKS,
                        "[mcbowagent] received action: seq={} d_yaw={} d_pitch={} fire_ok={}", seq, dy, dp, fo);
            }
        } catch (java.io.EOFException e) {
            // peer closed cleanly — normal exit
        } catch (IOException e) {
            // socket closed by writer cleanup or peer reset — also normal exit during shutdown
            trace(actionSeq <= VERBOSE_TICKS, "[mcbowagent] reader exited: {}", e.toString());
        } catch (Exception e) {
            // Fatal parse error etc. Tear the client down (closeClient -> latestAction=null + interrupt
            // frameLock so the writer's takeFrame unblocks and its next write throws), otherwise the
            // writer would keep streaming frames into a void while the bot freezes on its last action.
            LOGGER.error("[mcbowagent] reader thread error — closing client", e);
            closeClient();
        }
    }

    private byte[] takeFrame() {
        synchronized (frameLock) {
            if (frameToSend == null) {
                try { frameLock.wait(1000); } catch (InterruptedException e) { return null; }
            }
            byte[] f = frameToSend;
            frameToSend = null;
            return f;
        }
    }

    private void closeClient() {
        Socket c = client;
        client = null;
        latestAction = null;       // req 3: never reuse a stale action after disconnect
        latestBoxes = null;
        lastActionTimeMs = 0;
        // wake the writer thread if it's blocked in takeFrame.wait so it observes client==null and exits
        synchronized (frameLock) {
            frameToSend = null;
            frameLock.notifyAll();
        }
        if (c != null) {
            try { c.close(); } catch (IOException ignored) {}
        }
    }

    // ---------------- main thread: capture + apply control ----------------

    public void controlTick(MinecraftClient mc) {
        if (!cfg.runtimeActive) return;
        ClientPlayerEntity p = mc.player;
        if (p == null) return;

        // req 4: no Python client -> no control at all
        if (client == null) {
            controller.stopUsing(mc);
            lastAppliedActionSeq = -1;
            lastAligned = false;
            prevAligned = false;
            overlay(mc, "mcbow: no client");
            return;
        }

        controlTicks++;

        // Request a capture every N ticks; the actual GL readback happens on the render thread in
        // renderCapture(), BEFORE our ESP overlay is drawn, so our detection boxes never bake into the
        // frame the detector sees (which would feed back and corrupt the bbox/range). A failed/missing
        // capture just means no new frame -> the action goes stale -> the bow is stopped below.
        int interval = Math.max(1, cfg.runtimeCaptureInterval);
        if (controlTicks % interval == 0) captureRequested = true;

        // Single volatile read -> consistent snapshot (atomic publish).
        RuntimeAction act = latestAction;
        long now = System.currentTimeMillis();
        boolean fresh = act != null && (now - act.timeMs) <= STALE_MS;
        boolean hasTarget = fresh && act.hasTarget;

        if (hasTarget) {
            double dYaw = act.dYaw;
            double dPitch = act.dPitch;
            boolean fireOk = act.fireOk;

            // turn ONCE per NEW action (no spin); the next frame reflects the turn. Use act.seq (from the
            // SAME snapshot we just read above) so the seq matches the dYaw/dPitch we'll apply.
            long seq = act.seq;
            if (seq != lastAppliedActionSeq) {
                lastAppliedActionSeq = seq;
                prevAligned = lastAligned;               // remember the prior applied frame's alignment
                lastAligned = controller.stepView(p, p.yaw + dYaw, p.pitch + dPitch);
            }
            if (fireOk) {
                lastFireableMs = now;
                // fire only on TWO consecutive aligned frames (a single jittered <2deg frame can't loose).
                controller.manageBow(mc, lastAligned && prevAligned);
                overlay(mc, String.format("mcbow: yaw%+.1f pitch%+.1f FIRE", dYaw, dPitch));
            } else {
                controller.stopUsing(mc);                 // in sight but out of range: track only, no draw
                overlay(mc, String.format("mcbow: yaw%+.1f pitch%+.1f (out of range)", dYaw, dPitch));
            }
        } else if (now - lastFireableMs <= GRACE_MS) {
            // brief detection gap: keep the bow charging at the last aim so a shot can finish.
            controller.manageBow(mc, lastAligned && prevAligned);
            overlay(mc, "mcbow: grace (holding shot)");
        } else {
            // nothing for a while: drop the bow and ease the view back toward HOME (the F7 orientation) +
            // the horizon, so after clearing one side it TURNS BACK to the arena / the other side.
            controller.stopUsing(mc);
            controller.stepView(p, haveHome ? homeYaw : p.yaw, 0.0);
            lastAligned = false;
            prevAligned = false;
            overlay(mc, "mcbow: search (returning home)");
        }
    }

    /** Render-thread hook: if a capture was requested this tick, do the GL readback NOW — called from the HUD
     *  callback BEFORE our ESP overlay is drawn, so the captured frame keeps the vanilla HUD (train/infer
     *  parity) but NOT our boxes. No-op when idle / no client / nothing requested. */
    public void renderCapture(MinecraftClient mc) {
        if (!cfg.runtimeActive || client == null || !captureRequested) return;
        captureRequested = false;
        doCapture(mc);
    }

    private void doCapture(MinecraftClient mc) {
        boolean v = controlTicks <= VERBOSE_TICKS;
        try {
            int w = mc.getWindow().getScaledWidth();
            int h = mc.getWindow().getScaledHeight();
            trace(v, "[mcbowagent] before capture {}x{}", w, h);
            byte[] payload = cfg.runtimeRawFrame
                    ? frameCapture.captureRawBytes(mc, w, h)   // fast path: raw BGR, no PNG encode
                    : frameCapture.captureBytes(mc, w, h);     // legacy PNG path (still supported)
            captureFailStreak = 0;                   // a successful capture clears the failure streak
            if (payload == null) {
                trace(v, "[mcbowagent] capture returned null (framebuffer not ready?)");
                return;
            }
            // dedup via CRC32 — cheaper than Arrays.equals on 300 KB and avoids holding a hard ref to the last frame
            dedupCrc.reset();
            dedupCrc.update(payload, 0, payload.length);
            long crc = dedupCrc.getValue();
            if (haveLastSentCrc && crc == lastSentCrc) {
                trace(v, "[mcbowagent] duplicate frame (crc match) - skip send");
                return;
            }
            lastSentCrc = crc;
            haveLastSentCrc = true;
            trace(v, "[mcbowagent] capture success: bytes={} {}x{} fmt={}",
                    payload.length, w, h, cfg.runtimeRawFrame ? "raw" : "png");
            synchronized (frameLock) {
                frameToSend = payload;
                frameLock.notifyAll();
            }
        } catch (Exception e) {                      // transient GL/encode hiccup: skip the frame, keep the client
            captureFailStreak++;
            trace(captureFailStreak <= 3, "[mcbowagent] capture exception (#{}): {}", captureFailStreak, e.toString());
            if (captureFailStreak >= CAPTURE_FAIL_LIMIT) {
                LOGGER.error("[mcbowagent] capture failing persistently (" + captureFailStreak
                        + "x) - dropping client to recover", e);
                closeClient();
                captureFailStreak = 0;
            }
        }
    }

    // ---------------- helpers ----------------

    private void overlay(MinecraftClient mc, String msg) {
        if (mc.inGameHud != null) {
            mc.inGameHud.setOverlayMessage(new LiteralText(msg), false);
        }
    }

    private void trace(boolean verbose, String msg, Object... args) {
        if (verbose) LOGGER.info(msg, args);
        else LOGGER.debug(msg, args);
    }

    private static double optD(JsonObject o, String k) {
        return o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsDouble() : 0.0;
    }

    private static boolean optB(JsonObject o, String k) {
        return o.has(k) && !o.get(k).isJsonNull() && o.get(k).getAsBoolean();
    }

    /** Parse the action's "boxes":[[x0,y0,x1,y1,role],...] into an int[][], or null on absence/malformed. */
    private static int[][] parseBoxes(JsonObject o) {
        try {
            JsonElement be = o.get("boxes");
            if (be == null || !be.isJsonArray()) return null;
            JsonArray arr = be.getAsJsonArray();
            int[][] bx = new int[arr.size()][];
            for (int i = 0; i < arr.size(); i++) {
                JsonArray e = arr.get(i).getAsJsonArray();
                bx[i] = new int[]{e.get(0).getAsInt(), e.get(1).getAsInt(), e.get(2).getAsInt(),
                        e.get(3).getAsInt(), e.get(4).getAsInt()};
            }
            return bx;
        } catch (Exception ex) {
            return null;
        }
    }
}
