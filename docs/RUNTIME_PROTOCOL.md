# v1 运行时协议(mod ↔ Python)

锁步、localhost TCP。**mod = server**(游戏内常驻,开端口),**Python = client**(`runtime_loop.py` 连入)。每个 game tick 一来一回。

## 线路格式
两个方向都是:`4 字节大端 uint32 长度 + 负载`。
- **mod → Python**:负载 = 一帧的 **PNG 字节**,分辨率必须 = 检测器训练分辨率(`getScaledWidth/Height`,即 427×H),且**不含调试叠加**(录制时已隐藏;运行时同样别画)。
- **Python → mod**:负载 = UTF-8 JSON 动作。

## 动作 JSON(Python 每帧回一个)
```json
{"has_target": true, "d_yaw": 12.3, "d_pitch": -1.8, "range": 7.4, "fire_ok": true, "n_det": 5}
```
- `d_yaw`/`d_pitch`:要转到瞄准点的**相对增量(度)**,相对当前朝向;`+d_yaw`=右转,`+d_pitch`=向下;已含落点补偿(向上抬)。**可能 >10°**。
- `range`:估计距离(格);`fire_ok`:在弓射程内;`has_target=false` 时其余字段忽略。

## mod 每 tick 的处理(复用现成 `BowMacroController`)
```
收到 action:
  若 has_target:
     目标角 = (玩家当前 yaw + d_yaw, 当前 pitch + d_pitch)
     BowMacroController.stepView(玩家, 目标yaw, 目标pitch)   // 已限幅 ≤10°/tick,自动分多次转
     若 fire_ok: 维持拉弓(use=按住);当 |d_yaw|<~2 且 |d_pitch|<~2 且 蓄力≥20 tick → 松弓(开火)
  否则:
     releaseBow();(可选)缓慢扫描旋转找目标
发送当前帧(framebuffer→下采样到 scaled 分辨率→PNG),等待下一个 action(锁步)
```

## 稳定性与调试(hardened)
- **网络 I/O 在独立线程**(`mcbowagent-net`),GL 抓帧 + 弓控制在主线程。主线程绝不阻塞在 socket 上 → Python 慢/断都不会冻游戏。net 线程读超时 5s(容忍首帧推理慢)。
- **`latestAction` + 200ms staleness**:控制只在动作新鲜时执行;无 client / 动作过期 / 无目标 → `stopUsing()`(停拉弓但**不发射**),actionbar 显示 `no client` / `stale Nms` / `no_action`。
- **抓帧异常**:打完整 stacktrace、关当前 client(Python 自动重连)、server 继续 accept、游戏不崩。
- **F7 开/关、断连**:都清空 `latestAction` 并 `stopUsing`,不残留旧动作。
- **latest.log 日志**(Log4J,前 30 帧 INFO 后转 DEBUG 防刷屏):server started / client accepted / before capture / capture success(bytes,w,h)/ send frame seq / received action(seq,d_yaw,d_pitch,fire_ok)/ 各类 exception stacktrace / client disconnected。**首帧 abort 看这段就能定位卡在哪一步。**
- **Python 端**:`runtime_loop` 自动重连(断了打印 `mod disconnected; reconnecting`,不退出);`--debug-protocol` 打印每帧 `[recv]`(seq/png_bytes/shape)和每动作 `[send]`(d_yaw/d_pitch/range/fire_ok)。

## 要点
- **锁步**:mod 发帧后阻塞等 action 再进下一 tick;`d_yaw` 是基于"发帧时"的朝向,锁步保证只差 ~1 帧,闭环每 tick 收敛(转角越来越小)。
- **开火时机在 mod 侧**(它知道蓄力 tick 数和对齐程度);Python 只给"瞄哪、能不能射"。
- Python 侧实现见 `python/mc_bow_agent/runtime_loop.py`,协议见 `protocol.py`,本地无游戏测试见 `selftest_loop.py`。
- mod 侧需新增:一个 socket server(`ServerSocket`),在 client tick 里发帧、收 action、调用上面的逻辑。`FrameCapture` 已能产 PNG(改成写入内存 byte[] 而非文件即可复用)。
