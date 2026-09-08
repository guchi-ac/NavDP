# NavDP 雷达日志与 BEV 障碍图诊断设计

## 目标

在不改变 NavDP 规划、MPC 求解和速度发布行为的前提下，为下一轮真机测试补齐
雷达证据：

- 保存每一帧原始 `/scan`；
- 将 RGB-D/规划记录关联到时间上最近的雷达帧；
- 从 `/scan` 独立生成定义明确的当前局部二值障碍图；
- 在现有 `*_mpc_rgb_bev.mp4` 中显示该障碍图和雷达新鲜度。

本次不读取 `/costmap` 或 `/costmap_local`，也不使用雷达结果筛选、修改或阻止
轨迹。

## 输入与坐标

客户端新增 `sensor_msgs/msg/LaserScan` 订阅，默认话题 `/scan`，使用 sensor
data QoS。实机探测结果为：

- 发布频率约 `10 Hz`；
- `1616` 条射线；
- 消息坐标系为 `laser_frame`；
- 无效回波主要编码为 `0.0`。

Mira3 URDF 中 `base_link -> laser_link` 的固定安装位姿为
`x=0.042 m, y=0, yaw=pi`。当前系统没有广播 `laser_frame` 对应 TF，因此客户
端显式使用这组平面外参，并要求消息 `frame_id` 与配置的
`--laser-frame=laser_frame` 一致。frame 不一致时拒绝该帧并限频报错，不猜测
变换。

新增参数：

```text
--scan-topic /scan
--laser-frame laser_frame
--laser-x 0.042
--laser-y 0.0
--laser-yaw 3.141592653589793
--scan-sync-slop 0.25
--scan-timeout 0.25
--laser-map-resolution 0.05
```

The Mira3 driver defines scan `-180 deg` as robot-forward and `0 deg` as
robot-rear, so the logical `laser_frame` requires a planar `pi rad` yaw into
`base_link`. The `0.25 s` synchronization window reflects the online driver measurement:
the newest laser header stamp trails RGB by `0.151–0.227 s`. Pairing still
selects the nearest scan inside that bounded window.

所有距离、角度和超时参数必须有限；地图分辨率和超时必须为正数。

## 雷达快照与同步

增加不可变 `LaserScanSnapshot`，保存：

- 单调递增 `sequence`；
- ROS `stamp_ns` 和本机 `received_at`；
- `frame_id`；
- `angle_min`、`angle_increment`、`range_min`、`range_max`；
- 原始 `ranges` 的只读 `float32` 副本；
- 雷达回调时最新的 odom 位姿。

客户端保留最近 50 帧雷达快照。RGB-D 回调以彩色图像 ROS 时间戳为基准，选择
时间差绝对值最小的雷达快照；只有差值不超过 `scan_sync_slop` 才写入
`FrameSnapshot`。这里不把 scan 加进 RGB-D 的
`ApproximateTimeSynchronizer`，避免雷达短暂丢帧阻断相机、NavDP 和现有
视频记录。

规划和控制行为不因雷达缺失或过期而改变。BEV 将其显示为
`LASER WAITING` 或 `LASER STALE`。

## JSONL 诊断格式

每次接受雷达消息后，立即向现有 `*_mpc.jsonl` 写入一条：

```json
{
  "type": "scan",
  "wall_time": 0.0,
  "monotonic_time": 0.0,
  "scan_sequence": 1,
  "stamp_ns": 0,
  "frame_id": "laser_frame",
  "angle_min": 0.0,
  "angle_increment": 0.0,
  "range_min": 0.1,
  "range_max": 16.0,
  "ranges": [],
  "odom": [0.0, 0.0, 0.0]
}
```

保留原始 `ranges`，包括 `0.0` 和非有限值；JSON 转换将非有限值写为 `null`，
保证文件仍是标准 JSON。

每条 `plan` 记录增加：

- `scan_sequence`；
- `scan_stamp_ns`；
- `scan_rgb_dt_s`：scan 与该 RGB 帧的 ROS 时间差；
- `scan_age_s`：规划完成时距离 scan 本机接收时间的年龄。

每条 `control` 记录增加当前最近 scan 的 `scan_sequence` 和 `scan_age_s`，
但不把完整 ranges 重复写入控制行。通过 `scan_sequence` 可以关联原始 scan
行、plan 行和 control 行。

## 当前局部障碍图

本次地图只表达当前雷达命中，不推断第三方 cost，也不做历史融合：

- 地图覆盖范围与现有 BEV 一致：前方 `6 m`、后方 `2 m`、左右各 `4 m`；
- 分辨率默认 `0.05 m`；
- 初值为 `0`，表示当前 scan 未检测到障碍；
- 每个有效回波终点对应栅格置为 `100`；
- `0.0`、NaN、Inf、小于 `range_min` 或大于 `range_max` 的数据忽略；
- 使用显式雷达外参把极坐标终点转换到 `base_link` 平面；
- 落在 Mira3 底盘矩形内部的点作为雷达自反射丢弃，矩形半尺寸采用
  `x=0.255 m, y=0.260 m`；
- 本次不做 footprint 膨胀，不将地图用于安全判断。

地图以纯 NumPy 函数生成，输出栅格、有效命中点和丢弃计数，供单元测试、
日志摘要和 BEV 共用。

## BEV 图层

`render_mpc_rgb_bev` 增加可选的 `laser_obstacle_xy`、`laser_status` 和
`laser_age_s` 输入。

绘制顺序：

1. 中轴线；
2. RGB-D 点云；
3. 当前雷达障碍点，以亮洋红色小圆点显示；
4. actual、MPC、selected、guide；
5. 底盘和文字。

雷达图层不遮盖轨迹。图例增加 `laser`，状态显示
`LASER OK age=... points=...`、`LASER WAITING` 或 `LASER STALE`。
雷达缺失或过期时不绘制旧障碍点，避免视频把历史点误当成当前障碍。

若配对 scan 带 odom 位姿，则先把雷达命中转换到 odom，再转换到该 RGB-D
快照对应的底盘坐标，消除 scan 与相机之间机器人运动造成的平移和偏航差；
odom 缺失时仅在 `scan_rgb_dt_s <= scan_sync_slop` 时使用 scan 时刻
`base_link` 点。

## 错误处理

- scan frame 不匹配：拒绝消息并限频报错；
- scan 字段长度或数值元数据非法：拒绝消息并记录错误；
- JSONL 写入失败：沿用现有诊断失败策略，不影响控制；
- 雷达地图生成失败：BEV 显示 `LASER ERROR`，不影响现有 RGB-D BEV；
- 雷达缺失或过期：不重用旧障碍点；
- 客户端退出时仍由现有 `JsonlWriter.close()` 和 MP4 封尾流程收尾。

## 测试与验收

纯函数测试：

- 极坐标射线使用 `x=0.042 m` 外参正确转换到底盘坐标；
- `0.0`、NaN、Inf 和越界 range 被忽略；
- 底盘矩形内自反射被删除；
- 障碍端点落入正确 `0.05 m` 栅格；
- scan 与 RGB 时间最近邻匹配及 slop 边界；
- 带 odom 的 scan 点正确变换到 RGB-D 快照底盘坐标。

渲染测试：

- 雷达点使用独立洋红颜色；
- 轨迹最后绘制并覆盖雷达点；
- WAITING/STALE 时不绘制旧雷达点；
- 图例和状态文字存在。

客户端测试：

- 订阅 `/scan` 并保存原始 ranges；
- scan、plan 和 control 通过 `scan_sequence` 关联；
- 新参数默认值和校验；
- 雷达状态不进入 `control_stop_reason`，不改变 `/cmd_vel`。

本地验证运行相关 unittest、Python 编译和 `git diff --check`。真机先以
dry-run 启动，验收最新 JSONL 含连续 scan 行、plan 可关联 scan，且
`*_mpc_rgb_bev.mp4` 中雷达障碍与现场障碍方向一致。确认诊断正确前不启用
任何雷达轨迹调整。
