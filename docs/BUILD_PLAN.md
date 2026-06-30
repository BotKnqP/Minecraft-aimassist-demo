# 落地清单（弓箭-only，自建 Fabric 路线）

对齐框架 §15 的"感知-策略联合训练顺序 + 噪声从第一天"。✅=已起步。

## 阶段 0 — 基础设施
- [x] 项目脚手架 + 已核实常数 `constants.py`
- [x] 世界坐标→VPT 离散相机动作 `action_mapping.py`（+ 自测）
- [x] tick 级弹道仿真/求解 `ballistic.py`
- [x] 录制记录契约 `data_schema.py`
- [x] `numpy` 装好、`selftest` 通过（6/6）
- [x] `mod/` Fabric 1.16.5 Loom 工程**骨架已生成**（23 文件；用 IntelliJ 引导 wrapper；未编译）
- [ ] git init

## 阶段 1 — M0 物理对齐门（任何训练之前）
- [ ] Fabric mod 导出每-tick 怪物真值 + AABB→屏幕 bbox（先在 HUD 画框肉眼校验）
- [ ] 把 `ballistic.solve_pitch/lead_target` 接成游戏内 aimbot，**发真箭验证命中**
- [ ] 发 10k 箭跨距离/蓄力/角度，断言**预测命中 tick == 实际 ±1**（校 drag/gravity/速度/tick 顺序）
- [ ] 测全链 frame→detect→solve→act 延迟（以 tick 计），建 latency 归因基线
- [ ] pitch 符号、相对-vs-绝对、不乘 CAMERA_SCALER 三个静默 bug 各写一条断言

## 阶段 2 — 录制（三套标签同步）
- [ ] 实现 FABRIC_MOD_SPEC 的 7 个组件，锁 20 tick/s
  - [x] 帧捕获（组件3）：framebuffer→下采样到 scaled 分辨率→PNG；录制时隐藏调试叠加（待 IDE 编译验证 Yarn 名）
- [ ] 编排 §15 场景（单怪居中 / 各边进入 / 身后 / 被包围 / 多距离高差 / 昼夜）
- [ ] 录 v0：zombie-only、superflat、活 60s，产出 `episode_*.jsonl` + 帧
- [ ] 写 `loader`：jsonl+帧 → 训练张量（policy / YOLO / estimator 三视图）

## 阶段 3 — 模型（阶段 2/3 并行）
- [x] YOLO 训练管线已搭并测：`dataset.py`(jsonl→YOLO 数据集,按 episode 切分/裁剪/背景帧,2/2 测) + `train.py`(Ultralytics 训练+ONNX 导出,`--check` 确认 4060 GPU 就绪);经 3-agent 对抗审查修复。**等真实帧后即可训。** v1 默认单类 zombie。
- [x] **训出 zombie YOLO（v8n）**：v1=单段1483帧(泄漏,mAP50≈0.62虚高);**v2=合并10段6921帧/31150框,不泄漏,40轮,真实 mAP50≈0.39 / mAP50-95≈0.17**,但**功能性强**(近/清晰僵尸 0.78-0.89 置信,部署 conf≈0.5),1.5ms/帧,`mcbow_zombie_v2/weights/best.onnx`。录制→数据集→训练→ONNX 全链路打通 ✅
- [ ] (可选)提升检测:imgsz↑(768/960)救远处小目标 / 多录不同段增多样性（待用户定）
- [x] **v1 脚本运行时大脑（B,已建+测）**：`aim.py`(选最近=最大框/conf≥0.5 → bbox 中心=方位 d_yaw/d_pitch → bbox 高=距离 → `ballistic` 落点补偿 → 开火判定,单测 5/5);`calibrate.py`(从真值拟合 距离=k/bbox高,k=244.3,中位误差 10.5%/实战段 3-8%,31157 框);`runtime.py`(感知→决策,I/O 可插拔,真实帧验证产出合理瞄准指令)。
- [x] **B 集成 · Python 实时循环(已建+测)**：`protocol.py`(长度前缀帧;mod 推 PNG / Python 回动作 JSON)、`runtime_loop.py`(锁步:收帧→检测→选最近→回 {d_yaw,d_pitch,range,fire_ok})、`selftest_loop.py`(mock-mod 本地测,确定性 2/2 + **真 YOLO 过 socket 验证**:近战帧→7检出→动作合理)。协议见 `docs/RUNTIME_PROTOCOL.md`。
- [x] **B 集成 · mod 侧(已写,待 IDE 编译联机)**：`net/RuntimeBridge.java`(ServerSocket 锁步:推 PNG 帧→收 action JSON→`BowMacroController.stepView(当前角+delta)`+蓄力/松弓)、`FrameCapture.captureBytes`(内存 PNG)、`McBowAgentMod` 加 **F7** 开关 + tick 分流 + 运行时隐藏叠加、`RecorderConfig` 加 `runtimeActive/runtimePort`。Python 端加了连接重试。**B 代码完成,只差编译+联机点火。**
- [ ] state-estimator：先非神经（bbox 中心→yaw/pitch、尺寸→距离、帧差→速度），后 Kalman/小 MLP
- [ ] **测 YOLO 真实误差**（per-class 混淆、漏检 vs 距离/遮挡/光照、距离误差分布、延迟 tick）

## 阶段 4 — noisy-state policy（可部署）
- [ ] YOLO 真实回放：录的帧离线过 YOLO+estimator → 真实噪声 state，配干净专家动作
- [ ] 叠加按实测分布校准的合成 DR（角度/距离/误分类/漏检/假目标/延迟）
- [ ] 状态表示换 Set-Transformer + 有效性掩码（漏检=删 token）

## 阶段 5–6 — 联调 + DAgger
- [ ] Fabric env 包成最小 gym（socket 桥）供在线交互
- [ ] 端到端 RGB→YOLO→estimator→policy→action 联调
- [ ] DAgger：学生在感知环跑，aimbot 在失败态 relabel；"可实现 oracle"（看不见→重定位/扫描）
- [ ] maximin 选点跨 mob-mix/光照
