package com.mcbowagent.state;

import com.mcbowagent.record.JsonUtil;
import com.mcbowagent.vision.ProjectionUtil;

import net.minecraft.entity.Entity;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;

/** One hostile mob's privileged ground truth + its projected screen bbox. */
public final class MobSnapshot {

    public final String type;
    public final int entityId;
    public final double x, y, z;            // feet world pos
    public final double vx, vy, vz;         // blocks/tick
    public final float health;
    public final double relYaw, relPitch;
    public final double distance;
    public final double radialSpeed;        // + = approaching
    public final double tangentialSpeed;
    public final boolean visible;
    public final int[] bbox;                // {x0,y0,x1,y1} or null

    private MobSnapshot(String type, int entityId, double x, double y, double z,
                        double vx, double vy, double vz, float health,
                        double relYaw, double relPitch, double distance,
                        double radialSpeed, double tangentialSpeed,
                        boolean visible, int[] bbox) {
        this.type = type; this.entityId = entityId;
        this.x = x; this.y = y; this.z = z;
        this.vx = vx; this.vy = vy; this.vz = vz;
        this.health = health;
        this.relYaw = relYaw; this.relPitch = relPitch; this.distance = distance;
        this.radialSpeed = radialSpeed; this.tangentialSpeed = tangentialSpeed;
        this.visible = visible; this.bbox = bbox;
    }

    public static MobSnapshot from(Entity e, String type, float entHealth,
                                   Vec3d eye, float playerYaw, float playerPitch,
                                   Vec3d playerVel, boolean visible, int[] bbox) {
        // velocity from per-tick position delta (more reliable client-side than getVelocity()).
        double vx = e.getX() - e.prevX, vy = e.getY() - e.prevY, vz = e.getZ() - e.prevZ;
        Vec3d aim = new Vec3d(e.getX(), e.getY() + e.getHeight() * 0.5, e.getZ());

        double[] ang = ProjectionUtil.lookAngles(eye, aim);     // {yaw, pitch} deg, MC convention
        double relYaw = MathHelper.wrapDegrees(ang[0] - playerYaw);
        double relPitch = ang[1] - playerPitch;
        double dist = eye.distanceTo(aim);

        Vec3d toPlayer = eye.subtract(aim);
        Vec3d dir = toPlayer.lengthSquared() > 1e-9 ? toPlayer.normalize() : new Vec3d(0, 0, 0);
        Vec3d relVel = new Vec3d(vx, vy, vz).subtract(playerVel);
        double radial = relVel.dotProduct(dir);
        double tang = Math.sqrt(Math.max(0.0, relVel.lengthSquared() - radial * radial));

        return new MobSnapshot(type, e.getEntityId(), e.getX(), e.getY(), e.getZ(),
                vx, vy, vz, entHealth, relYaw, relPitch, dist, radial, tang, visible, bbox);
    }

    /** Full JSON object (array element under "mobs"). */
    public void writeObject(StringBuilder sb) {
        sb.append('{');
        sb.append("\"type\":").append(JsonUtil.str(type)).append(',');
        sb.append("\"entity_id\":").append(entityId).append(',');
        sb.append("\"world_xyz\":").append(JsonUtil.vec3(x, y, z)).append(',');
        sb.append("\"velocity\":").append(JsonUtil.vec3(vx, vy, vz)).append(',');
        sb.append("\"health\":").append(JsonUtil.num(health)).append(',');
        sb.append("\"rel_yaw\":").append(JsonUtil.num(relYaw)).append(',');
        sb.append("\"rel_pitch\":").append(JsonUtil.num(relPitch)).append(',');
        sb.append("\"distance\":").append(JsonUtil.num(distance)).append(',');
        sb.append("\"radial_speed\":").append(JsonUtil.num(radialSpeed)).append(',');
        sb.append("\"tangential_speed\":").append(JsonUtil.num(tangentialSpeed)).append(',');
        sb.append("\"visible\":").append(JsonUtil.bool(visible)).append(',');
        sb.append("\"screen_bbox\":");
        if (bbox == null) {
            sb.append("null");
        } else {
            sb.append('[').append(bbox[0]).append(',').append(bbox[1]).append(',')
              .append(bbox[2]).append(',').append(bbox[3]).append(']');
        }
        sb.append('}');
    }
}
