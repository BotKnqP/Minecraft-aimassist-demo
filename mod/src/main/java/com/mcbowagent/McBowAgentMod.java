package com.mcbowagent;

import com.mcbowagent.config.RecorderConfig;
import com.mcbowagent.net.RuntimeBridge;
import com.mcbowagent.oracle.BowAimbotOracle;
import com.mcbowagent.record.TickRecorder;
import com.mcbowagent.vision.HudBboxRenderer;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;

import org.lwjgl.glfw.GLFW;

/**
 * Client entrypoint. Uses Fabric API events (ClientTickEvents / HudRenderCallback)
 * rather than raw mixins for ticking + HUD 鈥?idiomatic and robust. The mixin
 * package is reserved for the framebuffer-capture mixin (FrameCapture TODO).
 *
 * Keys: F6 cycle resolution presets, F7 vision runtime, F8 start/stop recording,
 *       F9 toggle HUD bboxes, F10 toggle aimbot oracle.
 */
public class McBowAgentMod implements ClientModInitializer {

    public static final String MOD_ID = "mcbowagent";

    /** True while the vision runtime (F7) owns the view — read by MouseLookMixin to suppress physical
     *  mouse-look so it doesn't fight the bot's direct yaw/pitch control ("抢鼠标"). */
    public static volatile boolean RUNTIME_OWNS_VIEW = false;

    private final RecorderConfig config = new RecorderConfig();
    private final BowAimbotOracle oracle = new BowAimbotOracle();
    private final TickRecorder recorder = new TickRecorder(config, oracle);
    private final RuntimeBridge bridge = new RuntimeBridge(config);

    private KeyBinding keyToggleRecord;
    private KeyBinding keyToggleHud;
    private KeyBinding keyToggleOracle;
    private KeyBinding keyToggleRuntime;
    private KeyBinding keyCycleRes;

    /** F6-cycled fullscreen / window-size presets. First press DOWN-shifts (1280x720 is the most useful
     *  drop for getting >60 FPS at modest GPU), then progressively smaller, then wraps back to 1920x1080
     *  so the user can quickly return to "native-ish" without restarting the game. Verified by workflow:
     *  Sodium 0.2.0 has no Window/Framebuffer/GLFW mixin, so the vanilla onResolutionChanged path runs
     *  unmodified — no conflict. */
    private static final int[][] RES_PRESETS = {
            {1280, 720},
            {960, 540},
            {854, 480},
            {1920, 1080},
    };
    private int resPresetIdx = 0;

    @Override
    public void onInitializeClient() {
        keyToggleRecord = register("toggle_record", GLFW.GLFW_KEY_F8);
        keyToggleHud = register("toggle_hud", GLFW.GLFW_KEY_F9);
        keyToggleOracle = register("toggle_oracle", GLFW.GLFW_KEY_F10);
        keyToggleRuntime = register("toggle_runtime", GLFW.GLFW_KEY_F7);
        keyCycleRes = register("cycle_res", GLFW.GLFW_KEY_F6);

        ClientTickEvents.END_CLIENT_TICK.register(this::onEndTick);

        HudRenderCallback.EVENT.register((matrices, tickDelta) -> {
            MinecraftClient mc = MinecraftClient.getInstance();
            if (mc.player == null) return;
            if (config.runtimeActive) {
                // CAPTURE the detector's frame HERE, before our ESP is drawn -> the captured frame keeps the
                // vanilla HUD (train/infer parity) but NOT our boxes (otherwise they feed back into detection).
                bridge.renderCapture(mc);
                if (config.hudBboxes) {
                    // Use the RENDER camera's yaw/pitch (interpolated to THIS render frame via tickDelta)
                    // instead of player.yaw (tick-quantised at 20 Hz). At 60-300 fps the difference is up
                    // to half a tick of lag = up to ~25 px shift at 1080p / fov 70 — visible as "boxes
                    // can't keep up with the camera at large windows / high res".
                    net.minecraft.client.render.Camera cam = mc.gameRenderer.getCamera();
                    HudBboxRenderer.renderRuntime(matrices, mc, bridge.getLatestBoxes(),
                            bridge.getCaptureYaw(), bridge.getCapturePitch(),
                            cam.getYaw(), cam.getPitch(), bridge.hasCaptureView());
                }
                return;
            }
            if (!config.hudBboxes || config.recording) return;   // F9 toggle; clean frames while recording
            HudBboxRenderer.render(matrices, mc, recorder.getLatestMobs(),
                    config.recording, config.oracleEnabled, recorder.getTick());
        });

        System.out.println("[mcbowagent] initialized (F6 res-cycle, F7 runtime, F8 record, F9 hud, F10 oracle)");
    }

    private KeyBinding register(String name, int keyCode) {
        return KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.mcbowagent." + name, InputUtil.Type.KEYSYM, keyCode, "category.mcbowagent"));
    }

    private void onEndTick(MinecraftClient mc) {
        while (keyToggleRecord.wasPressed()) {
            if (config.recording) recorder.stopRecording();
            else recorder.startRecording(mc);
        }
        while (keyToggleHud.wasPressed()) {
            config.hudBboxes = !config.hudBboxes;
        }
        while (keyToggleOracle.wasPressed()) {
            config.oracleEnabled = !config.oracleEnabled;
            if (!config.oracleEnabled && mc.player != null) {
                mc.options.keyUse.setPressed(false);   // don't leave the bow drawn
            }
        }
        while (keyCycleRes.wasPressed()) {
            int[] wh = RES_PRESETS[resPresetIdx];
            resPresetIdx = (resPresetIdx + 1) % RES_PRESETS.length;
            int w = wh[0], h = wh[1];
            net.minecraft.client.util.Window win = mc.getWindow();
            net.minecraft.client.util.Monitor mon = win.getMonitor();
            if (mon != null) {
                int hz = mon.getCurrentVideoMode().getRefreshRate();
                net.minecraft.client.util.VideoMode target = mon.findClosestVideoMode(
                        java.util.Optional.of(new net.minecraft.client.util.VideoMode(w, h, 8, 8, 8, hz)));
                // Fullscreen path: setVideoMode flips dirty; applyVideoMode no-ops if !fullscreen.
                win.setVideoMode(java.util.Optional.of(target));
                win.applyVideoMode();
                mc.options.fullscreenResolution = target.asString();
                mc.options.write();           // persist so next launch matches
                // Windowed path: 1.16.5 has no Window.setWindowedSize. Drive GLFW directly; the engine's
                // onWindowSizeChanged callback recomputes framebuffer + GUI scale + fires onResolutionChanged.
                if (!win.isFullscreen()) {
                    GLFW.glfwSetWindowSize(win.getHandle(), target.getWidth(), target.getHeight());
                }
                System.out.println("[mcbowagent] resolution -> " + target.getWidth() + "x" + target.getHeight()
                        + (win.isFullscreen() ? " (fullscreen)" : " (windowed)"));
            } else {
                GLFW.glfwSetWindowSize(win.getHandle(), w, h);
                System.out.println("[mcbowagent] resolution -> " + w + "x" + h + " (windowed, no monitor)");
            }
        }
        while (keyToggleRuntime.wasPressed()) {
            config.runtimeActive = !config.runtimeActive;
            if (config.runtimeActive) {
                bridge.start(mc, config.runtimePort);
            } else {
                bridge.stop(mc);
            }
            RUNTIME_OWNS_VIEW = config.runtimeActive;   // suppress physical mouse-look while the bot aims
        }
        if (mc.player != null && mc.world != null) {
            if (config.runtimeActive) {
                bridge.controlTick(mc);     // vision -> bow, driven by the Python client
            } else {
                recorder.onClientTick(mc);
            }
        }
    }
}

