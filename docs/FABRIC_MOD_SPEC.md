# 录制器 Fabric mod 规格（MC 1.16.5）

一个 1.16.5 Fabric 模组，每 game tick（20/s）记录专家演示的**三套标签 + 事件**，并能编排怪物波次。输出严格对应 `python/mc_bow_agent/data_schema.py` 的 `TickRecord`。

## 构建环境
- Fabric Loader（1.16.5）+ fabric-api 0.42.0（已装）。
- Java 目标 = 8（用 Gradle toolchain 自动拉取；本机只有 JDK 21 也行）。
- 用 Fabric example mod 模板 + Loom 起 Gradle 工程，放在 `mod/`。

## 七个组件

1. **真值导出器**（每 tick）：遍历玩家附近的敌对怪，取 `type / entityId / world XYZ / velocity(=pos−prevPos) / health / AABB`。再算相对玩家的 `yaw、pitch、distance、radial_speed(接近为正)、tangential_speed`。
2. **AABB→屏幕投影**：用相机 yaw/pitch + projection 矩阵把怪物世界 AABB 的 8 角投到 NDC→像素，取 min/max 得 `screen_bbox`。数学参考 BoundingBoxOutlineReloaded / `Frustum`。`visible = 在视锥内 且 中心射线未被方块遮挡`。
3. **帧捕获**：进程内读 framebuffer（mixin 到渲染后的 `Framebuffer` 颜色附件），下采样到 128×128 存 PNG/JPEG。**不要用 OS 截屏**（慢、相机矩阵不精确）。
4. **专家动作捕获**：你的 aimbot 控制角色；每 tick 记录"它下达的动作"——`camera(d_pitch,d_yaw, 度)`（先过 `action_mapping.world_to_camera_action` 使其本就可 VPT 离散）、移动键、`use(蓄力 hold/release)`、选中槽、`target_entity_id`。记录**指令动作**而非渲染结果。
5. **波次控制**：服务端命令 `/summon`、`/effect <id> minecraft:speed`、`/time set night`、`/give @p arrow`，按脚本升级；`FastReset` 用 teleport 不重载世界。
6. **事件钩子**：弓箭释放（`use` 在蓄力≥3 tick 后松开）、箭命中实体、怪死亡、玩家受伤 → 填 `Events`。
7. **日志器**：每 tick 写一行 `TickRecord` JSON（`episode_{id}.jsonl`）+ 一张帧文件；每局一个文件夹。锁 20 tick/s。

## 关键正确性约束（与 §15 一致）
- `camera` 必须是**每 tick 相对增量、单位度**；single tick ≤ ±10°，大转角拆多 tick。
- 弓 = `use` 右键，持续 ≥18–20 tick 满蓄力再松；`attack` 不用于弓。
- pitch **向下为正**；导出真值与动作前用正下方目标验一遍符号。
- 录制时**不要**把度数乘 `CAMERA_SCALER`（env 字段已是度数）。
- 强制刷怪分布覆盖（全类型/全距离/全光照）+ 编排"怪从屏幕各边进入 / 身后 / 被包围"等 §15 场景。

## v0 范围（先打通闭环）
只 zombie、只 bow、superflat arena；先验证"同一局能同时产出 ① policy 的 action 标签 ② YOLO 的 bbox/class 标签 ③ estimator 的角度/距离标签 ④ 可 replay 的动作序列"。
