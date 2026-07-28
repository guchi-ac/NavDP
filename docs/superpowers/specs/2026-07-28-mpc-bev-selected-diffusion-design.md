# MPC BEV 原始 Selected Diffusion 可视化设计

## 目标

在 `mpc_rgb_bev.mp4` 中增加当前已安装 MPC 参考所对应的原始
selected diffusion 轨迹，使它能够与黄色 `active_traj` 引导点和红色
MPC 预测轨迹在同一帧、同一坐标系中直接比较。

这层可视化用于判断轨迹差异发生在 NavDP selected diffusion、
`TrajectoryManager`，还是 MPC 求解阶段。它不改变规划、轨迹管理、
MPC 求解或控制行为。

## 图层语义

BEV 路径图层从下到上为：

1. 绿色 `actual`：里程计实走轨迹。
2. 红色 `MPC`：当前 MPC 求解得到的预测状态。
3. 青色 `selected`：当前已安装 MPC 参考对应的原始 selected diffusion。
4. 黄色 `guide`：`TrajectoryManager.active_traj` 离散引导点。
5. 白色底盘矩形和方向箭头。

青色 selected 使用 2 px 抗锯齿连续折线。黄色 guide 点在青线上方绘制，
因此两者重合时仍能看清离散引导点；发生历史连接或轨迹管理差异时，青黄两层
会明确分离。

## 数据所有权与时序

规划线程从 NavDP 响应取得 selected diffusion，将其从相机平面坐标转换到
odom 坐标，得到 `retained_world_xy`。只有当
`TrajectoryManager` 接纳该候选并成功安装进 MPC 时，客户端才把这条原始
selected diffusion 保存为当前已安装 selected。

候选因 critic、连接距离或连接航向被拒绝时，不发布未采用的新青线。客户端
继续保留与当前黄色 guide 和 MPC 参考相对应的上一次已安装 selected，避免
把“最新生成但未执行”的候选误画成当前控制参考。

只有在候选已被接纳且 MPC 安装成功后，才更新已安装 selected。安装失败时
保留旧值；同时把已接纳候选作为 pending provenance 保留下来。后续即使新
候选被拒绝，只要由该 pending 候选延续出的 active trajectory 成功安装，
就把 pending 提升为 installed。`active_traj` 耗尽且控制参考变为不可用时
同时清空 pending 和 installed。

控制线程在每次 MPC 求解成功后，同时复制以下只读数据到同一个
`MpcVisualizationSnapshot`：

- MPC 预测状态；
- 已安装 `active_traj`；
- 已安装原始 selected diffusion；
- 当前命令、求解耗时和更新时间。

视频线程只读取这一个快照，从而保证青、黄、红三层属于同一次 MPC 状态，
不会从规划线程异步读取不同帧的数据。

## 坐标变换与新鲜度

已安装 selected 以 odom 坐标保存。渲染器使用与黄色 guide、红色 MPC
相同的 `world_xy_to_current_base()` 将它转换到当前 RGB-D 帧的底盘坐标，
再使用同一个 `_base_xy_to_pixels()` 投影到 BEV。

只有 MPC 快照新鲜且 odom 状态为 `ODOM OK` 时才显示青、黄、红三层。
快照或 odom 过期时三层一起隐藏。selected 为空时仅跳过青线，不影响其他
图层。

## 接口变化

`MpcVisualizationSnapshot` 增加只读的 `selected_diffusion` 数组。

客户端增加 `SelectedDiffusionInstallState` 两阶段状态：候选被接纳时
`stage` 为 pending，MPC 安装成功后 `commit` 为 installed。重建 MPC 和
原地更新 MPC 参考两条分支使用相同的更新规则。

`render_mpc_rgb_bev(...)` 增加可选参数 `selected_diffusion`。调用方传入
odom 坐标二维点；渲染器负责坐标转换、青色折线绘制和图例更新。

## 测试

- 渲染器像素测试验证青色 selected 连续线可见，并继续验证绿色、红色、
  黄色和白色图层。
- 客户端源代码测试验证 MPC 快照携带 selected，并传入 BEV 渲染器。
- 轨迹安装测试验证被接纳候选更新已安装 selected。
- 拒绝候选测试验证未采用的新候选不会覆盖已安装 selected。
- 失败重试测试验证已接纳候选在 MPC 安装失败后保留 pending provenance，
  并在后续历史轨迹安装成功时正确提升。
- 运行现有 RGB BEV、客户端和轨迹管理相关测试，确认没有控制行为回归。

## 非目标

- 不修改服务端 `fps_pointgoal` 可视化。
- 不显示全部 diffusion 候选。
- 不改变 `TrajectoryManager` 的接纳、历史连接或路径构造逻辑。
- 不改变 MPC 参考稠密化、代价函数、预测时域或速度输出。
- 不增加运行参数。
