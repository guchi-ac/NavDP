# NavDP 轮式真机 ROS2 客户端

这个客户端保留官方 NavDP 模型和 `/imagegoal_step` 服务，将 InternNav 的
RGB-D/里程计/MPC 真机外壳适配到 105 的 D435 与轮式底盘。

## 环境

GPU 主机运行官方 NavDP 服务，并把 105 的 `127.0.0.1:8888` 反向转发到
GPU 主机的 NavDP 端口。105 上执行：

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
export ROS_DOMAIN_ID=11
export ROS2CLI_NO_DAEMON=1
cd /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client
```

## Dry-run

Dry-run 会持续完成相机订阅、NavDP 请求、到达判断和规划，但不会发布底盘命令：

```bash
python3 scripts/realworld/navdp_imagegoal_client.py \
  --goal-image /path/to/goal.jpg
```

## 启用底盘

确认机器人周围无人、急停可用，并确保 Web Console、TMS 和 planning_node
没有主动发送导航命令，然后执行：

```bash
python3 scripts/realworld/navdp_imagegoal_client.py \
  --goal-image /path/to/goal.jpg \
  --enable-control
```

客户端仅在使用 `--enable-control` 时自动整姿：先通过
`/Torso/torso_action_service` 将本体 yaw 和头部 yaw 置为 `0°`，并将
`head_pitch: -19.7795845`。MIRA3 使用 `torso_mask=[false, true]`，因此保持
躯干高度不变、只控制 torso yaw；`head_mask=[true, true]` 同时控制 head yaw
和 head pitch。整姿最大速度为 `0.1`，等待超时可通过
`--posture-timeout 10.0` 修改。Dry-run 不创建或发送整姿 goal，不会移动本体
或头部。

整姿成功后，客户端才向 `/skill_behavior_tree` 发送与下面命令等价的
`local_nav` goal，确认 goal 被接受后才启动规划和控制线程：

```bash
ros2 action send_goal /skill_behavior_tree interfaces/action/Skill \
  '{"head": {"action_name": "local_nav" }}'
```

MPC 速度默认发布到 `LocalNavAction` 实际订阅的 `/cmd_vel`。客户端退出时先
连续发布三次零速度，再取消 `local_nav` goal。整姿或 local_nav action server
不存在、goal 被拒绝、超时或返回失败时，客户端直接退出，不会启动底盘控制。
Dry-run 不发送这两个 goal。

MPC 默认限制为 `--max-v 0.15 m/s` 和 `--max-w 0.50 rad/s`，并原样发布到
`/cmd_vel`。MPC 使用 `/odom` 位姿中的 `[x, y, yaw]` 作为状态，并将线速度和
角速度 `[v, w]` 作为控制量。MPC 从采样 guide 的局部切线生成参考 yaw，并使用
wrap 后的 yaw 误差。急转时优化器可降低 `v`、提高 `|w|`；没有基于 heading
误差的额外速度约束，也没有额外的 rotate-first 状态机。
RGB-D 几何验证器第一帧进入
候选到达状态时立即输出零速度，连续三帧后锁存到达。相机、里程计、规划、
有效执行轨迹或 MPC 异常也会输出零速度。

## 上游 NavDP MPC 与 D435 重投影

客户端以
`InternRobotics/NavDP@bebb436a9856acbd6ed2a63234a99db6bac2fd3a`
的 MPC 结构为基础，保留 `N=15`、`T=0.1`、`ref_gap=3` 和
`R=diag([0.02, 0.15])`。为跟踪 guide 方向并匹配真机速度，本部署有意将
yaw 代价权重从上游的 `0` 改为 `5`，并将参考速度从上游默认的
`desired_v=0.5` 改为 `desired_v=--max-v`。客户端采用
`InternRobotics/InternNav@7a5c62400ac45b313d9b709c740b64191556a242`
的在线轨迹交接方式：第一条有效轨迹创建 MPC，后续有效 selected diffusion
只更新同一个 MPC 的 `ref_traj`，不清空上一次最优解的 warm start。控制线程
发布完当前求解结果后，规划线程再原子更新参考轨迹；有效新规划不会在两个
控制周期之间额外插入零速度。不保存历史引导点、不补“底盘到首点”的盲区点，
也不对输入引导点重新采样。

NavDP 原始 selected diffusion 使用真机
`/cam_head/d435/color/camera_info` 发布的彩色相机内参，并按官方 RGB
记录方式固定采用水平相机和相机坐标地面 `Z=-0.2 m`。控制轨迹不使用真机
D435 的安装高度、平移、俯仰、横滚或偏航；黄色 `active_traj` 保留全部
官方几何转换后的 diffusion 点并直接送给 MPC。青色 selected 使用相同的
官方平面方向，仅用于对比。

真机 D435 optical TF 仍用于 RGB-D BEV 点云反投影和 TF 诊断，不参与
selected 到黄色 guide 的控制几何。官方控制高度固定为 `0.2 m`，没有运行时
覆盖参数。

如果射线与地面平行、交点位于相机后方、交点不在底盘前方，或者 critic
低于阈值，本轮规划会清空旧 MPC 和旧执行轨迹并停车，不会继续沿用上一次
路径。与 NavDP 官方实现一致，diffusion 点列不要求前向距离逐点单调。

上游 MPC 的 `N=15` 表示 15 个预测控制步，时间步长 `T=0.1 s`；它不表示
diffusion 引导点数量，也不会截断 diffusion 点。完整路径先按上游实现线性
加密 50 倍，再从离底盘最近的位置起按目标弧长选参考点。`ref_gap=3` 表示
每 3 个 MPC 步施加一次位置参考代价，因此 MPC 使用
`N // ref_gap + 1 = 6` 个参考状态。MPC 的参考速度使用
`desired_v=--max-v`，使参考点间距与可发布速度一致；在默认值下，相邻目标
参考弧长约为 `0.15 * 3 * 0.1 = 0.045 m`。角速度仍受 `--max-w 0.50` 限制。

红色轨迹表示实际交给 MPC 的 `active_traj`；按 critic 着色的轨迹仍表示模型
候选，便于观察重投影前后的差异。

## NavDP-only D435 TF

头部和躯干保持锁定时，只保留 NavDP 需要的相机树：

```text
base_link
`-- d435_link                         独立静态安装边
    |-- d435_color_frame              D435 驱动
    |   `-- d435_color_optical_frame  NavDP 查询 frame
    `-- d435_depth_frame              D435 驱动
        `-- d435_depth_optical_frame
```

机器人运行参数固定为：

```text
USE_URDF_UTILS=0
USE_REALSENSE_D435=1
USE_REALSENSE_D455=0
```

这是 NavDP 专用精简模式，会移除完整机器人 URDF、joint-state merger、
`robot_state_publisher` 和厂商 transform service。D435 驱动继续发布相机内部
TF；NavDP 客户端启动时发布固定的 `base_link -> d435_link` 安装边，并按
RGB 时间戳读取完整的 `base_link <- d435_color_optical_frame`。

该控制器的 pitch 命令符号与 TF 几何符号相反，因此客户端在启用控制时自动
使用 `-19.7795845°` 的目标，使相机光轴向下 `20°`，同时抵消 D435 零位约
`0.2204°` 的光轴偏差。不再需要提前手工发送 Torso action。

头部存在偏心转轴，所以客户端固化的安装边同时包含平移和旋转。无需另起
`static_transform_publisher`；启动客户端即可发布该 TF，退出客户端后该发布者
也随之退出。运行期间不要移动头部或躯干。

每个 RGB-D 快照按 RGB 消息时间戳查询
`base_link <- d435_color_optical_frame`。TF 缺失时丢弃该帧并保持底盘停止：

```bash
ros2 run tf2_ros tf2_echo base_link d435_color_optical_frame
```

## 后台运行

首次验收保持 dry-run，不加 `--enable-control`：

```bash
cd /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client
: > client.log
tmux new-session -d -s navdp_imagegoal_client \
  "bash -lc 'source /opt/ros/humble/setup.bash; \
source /home/dev/midea_humanoid_robot/install/setup.bash; \
export ROS_DOMAIN_ID=11 ROS2CLI_NO_DAEMON=1 PYTHONUNBUFFERED=1; \
cd /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client; \
exec python3 scripts/realworld/navdp_imagegoal_client.py \
--goal-image goal_far.jpg >> client.log 2>&1'"
```

查看状态：

```bash
tmux has-session -t navdp_imagegoal_client
tail -n 80 /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client/client.log
```

停止：

```bash
tmux send-keys -t navdp_imagegoal_client C-c
```

客户端会先停止规划、控制和可视化线程，再调用服务端
`POST /navigator_close`。日志出现 `NavDP closed: status=closed` 后，当前
MP4 已完成封尾，可以正常打开。不要用 `kill -9` 结束客户端或服务端。

## 实时可视化

NavDP 官方没有语义分割输出。这里复用官方真机评测的
`VisualizationManager`，画面包含当前 RGB、固定目标图、红色选中轨迹，以及
按 critic 值着色的全部候选轨迹。客户端显示和客户端 MP4 不再绘制 D435
深度占据图。

显示线程只保留最新 RGB-D 帧，默认把显示输入降采样到 `320x240`，生成
`880x880` 组合图，并将显示刷新上限设为 `15 FPS`。这些设置不改变 NavDP
推理、到达判断和底盘控制使用的原始 `640x480` 数据：

```text
--visualization-width 320
--visualization-fps 15
--opencv-threads 2
NAVDP_BLAS_THREADS=2
```

### 雷达诊断

客户端直接订阅 `/scan`，不读取 `/costmap` 或 `/costmap_local`。每帧原始
`LaserScan` 以 `type="scan"` 写入同一个 `*_mpc.jsonl`；`plan` 和
`control` 记录通过 `scan_sequence` 与雷达帧关联，`plan` 还保存 RGB 与雷达
时间戳之差 `scan_rgb_dt_s`。

当前二值图由客户端独立生成：`0` 表示本帧没有命中，`100` 表示本帧有效回波
终点。它不表示未知空间，也不继承任何第三方代价值。地图范围与 BEV 一致，
前方 `6 m`、后方 `2 m`、左右各 `4 m`，默认分辨率 `0.05 m`。零值、非有限
值、量程外数据和落在底盘自身矩形内的回波会被忽略。

MPC RGB BEV 中洋红点表示与 RGB 帧时间最近且仍新鲜的雷达命中。状态为
`LASER WAITING`、`LASER STALE` 或 `LASER ERROR` 时不绘制旧点。洋红雷达层
位于 selected、guide 和 MPC 轨迹下方，便于直接观察当前轨迹是否穿过障碍。

相关参数及默认值：

```text
--scan-topic /scan
--laser-frame laser_frame
--laser-x 0.042
--laser-y 0.0
--laser-yaw 0.0
--scan-sync-slop 0.10
--scan-timeout 0.25
--laser-map-resolution 0.05
```

其中 `laser-x/y/yaw` 是已核对的 Mira3 平面激光外参。本版本只记录和显示
雷达，不改变 NavDP selected、MPC、控制停止条件或 `/cmd_vel`。

客户端持续发布 `/navdp/visualization`，并原子更新：

```text
/home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client/latest_visualization.jpg
```

客户端还会把同一套无深度画面录制到：

```text
/home/dev/NavDP-official-bebb436/navdp_visualizations/YYYYMMDD_HHMMSS_navdp_footprint.mp4
```

按 `Ctrl+C` 后日志出现 `visualization MP4 finalized` 才表示 MP4 已经封尾。
GPU 服务端官方 MP4 在 HTTP 响应返回前生成；客户端 MP4 记录 105 上发布的
实时组合画面。

客户端还会生成同时间戳的 `*_mpc_rgb_bev.mp4`。左上角分别显示发送到
`/cmd_vel` 的 `desired` 线速度/角速度和 `/odom` 返回的 `actual` 线速度/
角速度；里程计缺失或过期时实际速度显示为 `nan`。

MPC BEV 视频的图层从下到上包括绿色 `actual` 实走里程计、红色 `MPC` 预测
轨迹、青色 `selected` 原始 selected diffusion、黄色 `guide` 离散引导点，
以及最后绘制的白色底盘矩形和方向箭头。青线只在本轮候选通过 critic、
重投影并成功安装进 MPC 后更新；无效候选不会保留旧路径。青色轨迹使用 NavDP
原始平面坐标；黄色点一一对应虚拟相机像素经真实 D435 透视与地面求交后的
完整 diffusion 控制坐标，所有点使用相同尺寸。黄色点不是 MPC 内部加密后的
`ref_traj`，底盘位置也不会额外生成黄色锚点。青黄两层的分离只表示原始
selected diffusion 经透视重投影发生的变化。

画面还会沿机器人坐标系 `y=0` 绘制 1 px 深灰色中轴辅助线；它位于 RGB-D
点云和全部轨迹图层下方，不加入图例，并且在 odom 尚未就绪时仍会显示。

只有 MPC 快照新鲜且里程计为 `ODOM OK` 时才显示红色 MPC 预测、青色 selected
和黄色 guide；里程计过期时三者都会隐藏，避免在过期坐标系中显示路径。这些
odom 坐标系路径使用该 RGB-D 帧捕获时的里程计位姿变换到相机当前底盘坐标，
而不是渲染时更新的 live odom；live odom 的时间戳仍用于新鲜度判定。

相关渲染器和客户端源代码测试：

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_rgb_bev_visualizer test_wheeled_client_core -v
```

完整相关测试套件：

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_controllers \
  test_goal_capture \
  test_navigator_close.NavigatorCloseClientTests \
  test_rgb_bev_visualizer \
  test_wheeled_client_core -v
```

仓库完整发现测试中的 `NavigatorCloseServerTests` 仍依赖现有环境未提供的定制
`baselines/navdp/navdp_server.py`，因此上述套件只运行不依赖该文件的
`NavigatorCloseClientTests`。

## MPC 诊断日志

客户端每次启动都会自动创建与 MP4 使用同一时间戳的 JSONL：

```text
/home/dev/NavDP-official-bebb436/navdp_logs/YYYYMMDD_HHMMSS_mpc.jsonl
```

`plan` 行记录 `selected_local_xy`、原始 `raw_selected_world_xy`、重投影后的
`reprojected_base_xy` / `reprojected_world_xy`、固定官方相机高度 `0.2 m`、
重投影状态与拒绝原因，并同时记录实际 `active_traj`、`mpc_horizon`、规划时
的 odom/相机位姿、critic 和规划耗时；其中 `camera_pose` 仅用于真机 TF 与
BEV 诊断，不影响控制轨迹投影。
`control` 行按控制周期记录 odom 位姿与实测速度、发送的 `v/w`、MPC 参考状态、
预测状态、求解耗时及 frame/odom/plan 数据年龄。其中
`desired_velocity=[linear_x, angular_z]` 是期望/发布速度，
`actual_velocity=[linear_x, angular_z]` 是里程计实测速度；旧字段 `command` 和
`odom_twist` 继续保留兼容。每行立即刷新，便于客户端异常退出后保留已写数据。
可用 `--mpc-log-dir` 修改输出目录。

在 105 上另开一个独立的浏览器查看服务：

```bash
tmux new-session -d -s navdp_visualization \
  "bash -lc 'source /opt/ros/humble/setup.bash; \
source /home/dev/midea_humanoid_robot/install/setup.bash; \
export ROS_DOMAIN_ID=11 ROS2CLI_NO_DAEMON=1 PYTHONUNBUFFERED=1; \
cd /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client; \
exec python3 scripts/realworld/navdp_visualization_server.py \
--port 8088 >> visualization.log 2>&1'"
```

浏览器访问：

```text
http://192.168.1.105:8088
```

页面上的 `拍摄 Goal` 按钮直接抓取 D435 原始彩色话题，不会保存缩放后的
NavDP 组合图。每次点击会保留一份时间戳图片，并原子更新固定路径：

```text
/home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client/goals/captured_goal_YYYYMMDD_HHMMSS_mmm.jpg
/home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client/goals/captured_goal_latest.jpg
```

下一次启动客户端时固定使用最新抓拍：

```bash
python3 scripts/realworld/navdp_imagegoal_client.py \
  --goal-image /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client/goals/captured_goal_latest.jpg
```

查看器对应参数为：

```text
--rgb-topic /cam_head/d435/color/image_raw
--goal-dir goals
```

检查 ROS 话题和查看器状态：

```bash
ros2 topic info /navdp/visualization
tmux has-session -t navdp_visualization
tail -n 80 /home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client/visualization.log
```

停止查看器：

```bash
tmux send-keys -t navdp_visualization C-c
```

105 当前 D435 彩色与对齐深度应使用相同的 30 FPS 配置：

```text
rgb_camera.color_profile: 640x480x30
depth_module.depth_profile: 640x480x30
```
