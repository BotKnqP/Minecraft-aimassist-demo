package com.mcbowagent.vision;

import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

/**
 * World -> screen projection and look-angle math.
 *
 * TODO(M0): this is a PINHOLE approximation using the vertical FOV from options.
 * The exact Minecraft projection is GameRenderer#getBasicProjectionMatrix combined
 * with the camera rotation quaternion; cross-check the bbox against the HUD overlay
 * and against MC's real matrices before trusting bboxes as YOLO labels. The screen-X
 * sign (left/right) in particular must be verified visually and flipped if mirrored.
 */
public final class ProjectionUtil {

    private ProjectionUtil() {}

    /** Minecraft look unit vector for (yaw, pitch) degrees (pitch DOWN-positive). */
    public static Vec3d lookVector(double yawDeg, double pitchDeg) {
        double yaw = Math.toRadians(yawDeg);
        double pitch = Math.toRadians(pitchDeg);
        return new Vec3d(
                -Math.sin(yaw) * Math.cos(pitch),
                -Math.sin(pitch),
                Math.cos(yaw) * Math.cos(pitch));
    }

    /** {yaw, pitch} degrees to look from eye at target (Minecraft convention). */
    public static double[] lookAngles(Vec3d eye, Vec3d target) {
        double dx = target.x - eye.x, dy = target.y - eye.y, dz = target.z - eye.z;
        double yaw = Math.toDegrees(-Math.atan2(dx, dz));
        double pitch = Math.toDegrees(-Math.atan2(dy, Math.sqrt(dx * dx + dz * dz)));
        return new double[]{yaw, pitch};
    }

    /** Project a world point to screen pixels. Returns {x, y} or null if behind camera.
     *  fovDeg is the VERTICAL field of view; w/h are the target raster size. */
    public static int[] worldToScreen(Vec3d world, Vec3d camPos, double camYaw, double camPitch,
                                      double fovDeg, int w, int h) {
        Vec3d fwd = lookVector(camYaw, camPitch);
        Vec3d worldUp = new Vec3d(0, 1, 0);
        Vec3d right = fwd.crossProduct(worldUp);
        if (right.lengthSquared() < 1e-9) right = new Vec3d(1, 0, 0);
        right = right.normalize();
        Vec3d camUp = right.crossProduct(fwd).normalize();

        Vec3d rel = world.subtract(camPos);
        double cz = rel.dotProduct(fwd);
        if (cz <= 0.05) return null;                 // behind / on the camera plane
        double cx = rel.dotProduct(right);
        double cy = rel.dotProduct(camUp);

        double f = 1.0 / Math.tan(Math.toRadians(fovDeg) / 2.0);
        double aspect = (double) w / (double) h;
        double ndcX = (cx / cz) * (f / aspect);
        double ndcY = (cy / cz) * f;
        int sx = (int) Math.round((ndcX * 0.5 + 0.5) * w);
        int sy = (int) Math.round((1.0 - (ndcY * 0.5 + 0.5)) * h);
        return new int[]{sx, sy};
    }

    /** Project an entity AABB to a screen bbox {x0,y0,x1,y1}, clamped to [0,w]x[0,h].
     *  Returns null if no corner is in front of the camera. */
    public static int[] projectAabb(Box box, Vec3d camPos, double camYaw, double camPitch,
                                    double fovDeg, int w, int h) {
        double[] xs = {box.minX, box.maxX};
        double[] ys = {box.minY, box.maxY};
        double[] zs = {box.minZ, box.maxZ};
        int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
        int maxX = Integer.MIN_VALUE, maxY = Integer.MIN_VALUE;
        int valid = 0;
        for (double cxw : xs) for (double cyw : ys) for (double czw : zs) {
            int[] p = worldToScreen(new Vec3d(cxw, cyw, czw), camPos, camYaw, camPitch, fovDeg, w, h);
            if (p == null) continue;
            valid++;
            if (p[0] < minX) minX = p[0];
            if (p[1] < minY) minY = p[1];
            if (p[0] > maxX) maxX = p[0];
            if (p[1] > maxY) maxY = p[1];
        }
        if (valid == 0) return null;
        minX = clamp(minX, 0, w); maxX = clamp(maxX, 0, w);
        minY = clamp(minY, 0, h); maxY = clamp(maxY, 0, h);
        if (maxX <= minX || maxY <= minY) return null;
        return new int[]{minX, minY, maxX, maxY};
    }

    private static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}
