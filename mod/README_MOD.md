# mcbowagent — Fabric 录制器 (MC 1.16.5)

每 tick 把"视觉帧路径 + 特权怪物真值 + 专家动作 + 事件"写成一行 JSON,字段严格对齐 `../python/mc_bow_agent/data_schema.py` 的 `TickRecord`。**仅单机/私有研究用,勿上公共服务器。**

## 构建

> ⚠️ 这套源码是骨架,**没有在本机编译过**(本机无 Gradle/IDE)。第一次构建请在 IDE 里解析依赖;若个别 1.16.5 Yarn 名称对不上(已在注释里标 TODO 的地方),IDE 会直接指出,改动很小。

**推荐:IntelliJ IDEA Community(免费)**
1. File → Open → 选 `D:\projects\mc-bow-agent\mod`。
2. 信任并加载 Gradle 工程;IDEA 会自动下载 Gradle 8.5 wrapper,Loom 会拉取 MC 1.16.5 + Yarn 映射,Java toolchain 会自动下载 JDK 8(本机只有 JDK 21 也行)。
3. Gradle 面板 → `build` → 产物在 `build/libs/mcbowagent-0.0.1.jar`(忽略 `-sources.jar`)。

**命令行(若已装独立 Gradle ≥8.5)**
```powershell
cd D:\projects\mc-bow-agent\mod
gradle wrapper --gradle-version 8.5   # 仅首次,生成 gradlew
.\gradlew build
```
> 没有独立 Gradle 也没关系——用 IDEA 即可,它自带 wrapper 引导。本仓库未附带 `gradle-wrapper.jar`(二进制),由上面任一方式生成。

## 安装进 HMCL
把 `build/libs/mcbowagent-0.0.1.jar` 复制到:
```
D:\文件\minecraftCVRL\.minecraft\mods\
```
然后用 HMCL 启动 **1.16.5-Fabric** 实例(已装 fabric-api 0.42.0+1.16)。

## 快捷键
| 键 | 作用 |
|---|---|
| **F7** | **开 / 关 视觉运行时**(socket 桥:把帧推给 Python、按其动作转向射击) |
| F8 | 开始 / 停止录制 |
| F9 | 开 / 关 HUD bbox 叠加(肉眼校验) |
| F10 | 开 / 关 弓箭 oracle(自动选靶+转向+蓄力射击) |

## 视觉运行时(B,联机用法)
1. 进游戏、拿好弓、面向竞技场。
2. 游戏内按 **F7**(mod 起 socket server),再启动 Python 端(连 `127.0.0.1:5555`,Python 会重试 ~60s,所以先开哪个都行):
   ```
   cd python
   python -m mc_bow_agent.runtime_loop --weights runs/detect/mcbow_zombie_v2/weights/best.pt --device cpu --imgsz 416
   ```
   - ⚠️ **运行时务必 `--device cpu`**:Minecraft 已占着 GPU,Python 再开 CUDA 上下文会把显存/提交内存挤爆(报 `Memory allocation failure` / cv2 `Insufficient memory`)。nano 模型 CPU 推理够快,锁步只是把 tick 节奏放慢一点。
   - 内存还紧:再降 `--imgsz 320`、在 HMCL 里调低 Minecraft 分配的 RAM、或关掉 Edge/杀软等大户。
   - `best.onnx` 想用就配 `--device cpu`(onnxruntime CPU);`--device cuda:0` 需要 cuDNN 9 + CUDA 13,本机是 CUDA 12.1,别用。
3. 连上后:mod 每 tick 把帧推给 Python,Python 回 `{d_yaw,d_pitch,fire_ok}`,mod 用 `BowMacroController` 转向+蓄力+开火。再按 F7 关闭。
4. 运行时会自动隐藏调试叠加(帧干净)。协议细节见 `../docs/RUNTIME_PROTOCOL.md`。
- 锁步阻塞在主线程(Python ~ms 应答);若 Python 没连/卡住,1 秒读超时后自动松弓、不冻游戏。

## 输出
```
D:\projects\mc-bow-agent\runs\run_YYYYMMDD_HHmmss\
  episode_0001.jsonl      # 每 tick 一行 TickRecord
  frames\frame_000001.png # 每 tick 抓帧,下采样到 bbox 分辨率(与标签同坐标系)
```
用 `python -m mc_bow_agent`(`data_schema.load_episode`)即可逐行读回。

## v0 范围 / 已知 TODO（都在源码里标注）
- **FrameCapture 已实现**:读主 framebuffer → 下采样到 bbox 的 scaled 分辨率 → 写 PNG;**录制时自动隐藏调试叠加**(绿框不入帧)。注意**原版 HUD(物品栏/准星)在帧内**(与运行时一致;想要纯净世界帧可在录制时按 F1 隐藏 HUD)。`NativeImage.resizeSubRectTo/loadFromTextureImage` 等是 1.16.5 Yarn 名,首次编译在 IDE 里确认。
- **ProjectionUtil 是 pinhole 近似**:用竖直 FOV;精确矩阵是 `GameRenderer#getBasicProjectionMatrix` + 相机旋转四元数。**先用 F9 HUD 肉眼核 bbox 是否框住怪、左右是否镜像(镜像就翻 screen-X 符号)。**
- **oracle 是直线瞄准**:暂不含 lead+drop。等 M0 物理门确认箭物理后,再把 `python/ballistic.py` 的 `solve_pitch/lead_target` 移植进来。
- **arrow_hit / kill 暂为 false**:需要箭实体/怪物血量事件钩子;`arrow_released`、`damage_taken` 已可用。
- **未自动选弓槽**:v0 假设手里已拿弓。

## 上手验证顺序（先别急着发 10k 箭）
1. mod 能加载进 1.16.5-Fabric(日志出现 `[mcbowagent] initialized`)。
2. F9 显示 HUD,状态行可见。
3. bbox 大致框住 zombie;左右不镜像(否则翻 `ProjectionUtil.worldToScreen` 的 screen-X 符号)。
4. F8 生成 `episode_0001.jsonl`。
5. 每行能被 Python `load_episode` 读取,字段名一致。
6. 玩家 yaw/pitch、怪物 distance / rel_yaw / rel_pitch 肉眼合理(正下方目标 pitch 为正)。
7. F10 oracle 能转向怪物,**不瞬移、不改实体位置**。
8. use 按住/释放被记录(`action.use`)。
9. `events.arrow_released` 在松弓时为 true。
10. 再进入 M0:发大量箭做物理对齐(预测命中 tick == 实际 ±1)。

## 设计取舍
tick 与 HUD 用 Fabric API 的 `ClientTickEvents.END_CLIENT_TICK` / `HudRenderCallback`(官方推荐、比裸 mixin 稳),所以没有 `ClientTickMixin/HudRenderMixin`;`com.mcbowagent.mixin` 包与 `mcbowagent.mixins.json` 留给以后真正需要注入的 framebuffer 抓帧。
