package com.mcbowagent.record;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import com.mcbowagent.config.RecorderConfig;
import com.mcbowagent.oracle.BowAimbotOracle;
import com.mcbowagent.state.EventSnapshot;
import com.mcbowagent.state.ExpertActionSnapshot;
import com.mcbowagent.state.MobSnapshot;
import com.mcbowagent.state.PlayerSnapshot;
import com.mcbowagent.vision.ProjectionUtil;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.render.Camera;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.item.Items;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;
import net.minecraft.util.registry.Registry;
import net.minecraft.world.RaycastContext;

/**
 * The per-tick heart: gathers the THREE label sets (frame path + privileged mob
 * truth + expert action) plus events and appends one JSONL row. Field names match
 * python/mc_bow_agent/data_schema.TickRecord exactly.
 */
public final class TickRecorder {

    private final RecorderConfig cfg;
    private final BowAimbotOracle oracle;
    private final FrameCapture frameCapture = new FrameCapture();

    private JsonlWriter writer;
    private AsyncFrameWriter pngWriter;     // off-thread PNG encode + disk write
    private Path runDir;
    private long tick = 0;

    // previous-tick trackers (for relative camera + event detection)
    private float prevYaw, prevPitch, prevHealth;
    private boolean prevUsingBow;
    private int prevBowCharge;
    private boolean hasPrev = false;

    // latest snapshots for the HUD overlay (rebuilt every tick, recording or not)
    private volatile List<MobSnapshot> latestMobs = new ArrayList<>();

    public TickRecorder(RecorderConfig cfg, BowAimbotOracle oracle) {
        this.cfg = cfg;
        this.oracle = oracle;
    }

    public List<MobSnapshot> getLatestMobs() { return latestMobs; }
    public long getTick() { return tick; }

    public synchronized void startRecording(MinecraftClient mc) {
        try {
            String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            Path dir = Paths.get(cfg.outputBaseDir, "run_" + stamp);
            Path episode = dir.resolve("episode_0001.jsonl");
            writer = new JsonlWriter(episode);
            pngWriter = new AsyncFrameWriter();
            this.runDir = dir;
            tick = 0;
            hasPrev = false;
            cfg.recording = true;
            System.out.println("[mcbowagent] recording -> " + episode);
        } catch (IOException e) {
            cfg.recording = false;
            System.err.println("[mcbowagent] failed to start recording: " + e.getMessage());
        }
    }

    public synchronized void stopRecording() {
        cfg.recording = false;
        if (writer != null) {
            try { writer.close(); } catch (IOException ignored) {}
            writer = null;
        }
        if (pngWriter != null) {
            long w = pngWriter.writtenCount();
            long d = pngWriter.droppedCount();
            pngWriter.close();        // drains the queue, joins the worker
            pngWriter = null;
            System.out.println("[mcbowagent] recording stopped at tick " + tick
                    + " (frames written=" + w + " dropped=" + d + ")");
        } else {
            System.out.println("[mcbowagent] recording stopped at tick " + tick);
        }
    }

    /** Call once per client tick (END_CLIENT_TICK). */
    public void onClientTick(MinecraftClient mc) {
        ClientPlayerEntity p = mc.player;
        ClientWorld world = mc.world;
        if (p == null || world == null) return;

        // 1) oracle drives the view/bow (only if enabled), returns the engaged target
        int targetId = cfg.oracleEnabled ? oracle.tick(mc, cfg) : -1;

        // 2) rebuild mob snapshots (for HUD + recording)
        Camera cam = mc.gameRenderer.getCamera();
        Vec3d camPos = cam.getPos();
        double camYaw = cam.getYaw();
        double camPitch = cam.getPitch();
        // bboxes are projected in SCALED-GUI pixels (getScaledWidth/Height). FrameCapture
        // MUST write frames at this exact resolution or the dataset labels will be invalid.
        double fov = mc.options.fov;
        int w = mc.getWindow().getScaledWidth();
        int h = mc.getWindow().getScaledHeight();

        Vec3d eye = p.getCameraPosVec(1.0F);
        Vec3d playerVel = new Vec3d(p.getX() - p.prevX, p.getY() - p.prevY, p.getZ() - p.prevZ);

        List<MobSnapshot> mobs = new ArrayList<>();
        for (Entity e : world.getEntities()) {
            if (!(e instanceof HostileEntity) || !e.isAlive()) continue;
            if (p.squaredDistanceTo(e) > cfg.scanRadius * cfg.scanRadius) continue;

            String type = Registry.ENTITY_TYPE.getId(e.getType()).getPath();
            float health = ((LivingEntity) e).getHealth();
            Vec3d aim = new Vec3d(e.getX(), e.getY() + e.getHeight() * 0.5, e.getZ());

            int[] bbox = ProjectionUtil.projectAabb(e.getBoundingBox(), camPos, camYaw, camPitch, fov, w, h);
            boolean onScreen = bbox != null;
            boolean occluded = isOccluded(world, p, eye, aim);
            boolean visible = onScreen && !occluded;

            mobs.add(MobSnapshot.from(e, type, health, eye, p.yaw, p.pitch, playerVel, visible, bbox));
        }
        this.latestMobs = mobs;

        // 3) write a row if recording. Sample at recordCaptureInterval (default 10 Hz) — the per-tick path
        //    is the heavy one (GL readback + PNG encode + disk write) and at 20 Hz the render thread chokes.
        int recInterval = Math.max(1, cfg.recordCaptureInterval);
        if (cfg.recording && writer != null && tick % recInterval == 0) {
            boolean curUsingBow = p.isUsingItem() && p.getActiveItem().getItem() == Items.BOW;
            int curCharge = curUsingBow ? p.getItemUseTime() : 0;
            float curHealth = p.getHealth();

            float pYaw = hasPrev ? prevYaw : p.yaw;
            float pPitch = hasPrev ? prevPitch : p.pitch;

            PlayerSnapshot player = PlayerSnapshot.from(p);
            ExpertActionSnapshot action = ExpertActionSnapshot.capture(mc, p.yaw, p.pitch, pYaw, pPitch, targetId);

            boolean arrowReleased = hasPrev && prevUsingBow && !curUsingBow
                    && prevBowCharge >= RecorderConfig.MIN_FIRE_TICKS;
            boolean damageTaken = hasPrev && curHealth < prevHealth;
            // TODO: arrow_hit / kill need entity-event hooks (track arrow entities or
            // hostile health drops / removals); v0 leaves them false.
            EventSnapshot events = new EventSnapshot(arrowReleased, false, false, damageTaken);

            String framePath = frameCapture.frameRelPath(tick);
            writeRow(player, mobs, action, events, framePath);
            if (runDir != null && pngWriter != null) {
                // GL readback + downscale ON the render thread (we have to — GL state is here), but PNG
                // encode + disk write GO TO the background writer thread. Frees the render thread of the
                // ~10-20 ms PNG/IO stall per recorded frame.
                net.minecraft.client.texture.NativeImage small = frameCapture.captureSmallImage(mc, w, h);
                if (small != null && !pngWriter.submit(small, runDir.resolve(framePath))) {
                    // queue full (disk stalled) — drop this frame to keep the game smooth
                    small.close();
                }
            }
        }
        if (cfg.recording && writer != null) tick++;   // advance the tick index every game tick while recording
                                                       // (so the recInterval modulo above samples correctly)

        // 4) update prev trackers every tick
        prevYaw = p.yaw;
        prevPitch = p.pitch;
        prevHealth = p.getHealth();
        prevUsingBow = p.isUsingItem() && p.getActiveItem().getItem() == Items.BOW;
        prevBowCharge = prevUsingBow ? p.getItemUseTime() : 0;
        hasPrev = true;
    }

    private boolean isOccluded(ClientWorld world, Entity viewer, Vec3d eye, Vec3d aim) {
        RaycastContext ctx = new RaycastContext(eye, aim,
                RaycastContext.ShapeType.COLLIDER, RaycastContext.FluidHandling.NONE, viewer);
        BlockHitResult hit = world.raycast(ctx);
        if (hit.getType() == HitResult.Type.MISS) return false;
        return eye.distanceTo(hit.getPos()) < eye.distanceTo(aim) - 0.1;
    }

    private void writeRow(PlayerSnapshot player, List<MobSnapshot> mobs,
                          ExpertActionSnapshot action, EventSnapshot events, String framePath) {
        StringBuilder sb = new StringBuilder(640);
        sb.append('{');
        sb.append("\"tick\":").append(tick).append(',');
        sb.append("\"frame_path\":").append(JsonUtil.str(framePath)).append(',');
        player.writeFields(sb);
        sb.append(',');
        sb.append("\"mobs\":[");
        for (int i = 0; i < mobs.size(); i++) {
            if (i > 0) sb.append(',');
            mobs.get(i).writeObject(sb);
        }
        sb.append("],");
        sb.append("\"action\":");
        action.writeObject(sb);
        sb.append(',');
        sb.append("\"events\":");
        events.writeObject(sb);
        sb.append('}');
        try {
            writer.writeLine(sb.toString());
        } catch (IOException e) {
            System.err.println("[mcbowagent] write failed: " + e.getMessage());
        }
    }
}
