# NavDP RGB-D ImageGoal 到达验证器

这个节点只验证“当前 D435 画面是否已经接近目标图像的拍摄位置”。它不会调用
NavDP，也不会发布底盘速度。

## 原理

1. 启动时读取并固定使用 `--goal-image`。
2. 订阅 D435 彩色图、对齐深度和相机内参。
3. 使用 SIFT 匹配当前图和目标图。
4. 使用当前深度把匹配点恢复为三维点。
5. 使用 `solvePnPRansac` 估计当前相机到目标相机的相对距离，并应用
   `--distance-scale` 标定系数。
6. 距离不大于 `--arrival-distance` 且连续多帧通过时，输出 `arrived=True`。

当前设备实测 `0.60 m` 时原始 PnP 输出约 `0.06 m`，因此节点默认使用
`--distance-scale 10.0`。如果重新标定，系数应按
`实测相机位移 / 原始 distance_m` 计算。匹配数量、内点数量和重投影误差是初始
工程参数，必须通过真机正负样本继续标定。

## 启动

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
export ROS_DOMAIN_ID=11
export ROS2CLI_NO_DAEMON=1

cd /home/dev/navdp_runtime/rgbd-goal-verifier
python3 scripts/realworld/rgbd_goal_verifier_node.py \
  --goal-image /path/to/goal.jpg \
  --distance-scale 10.0
```

默认话题：

```text
/cam_head/d435/color/image_raw
/cam_head/d435/aligned_depth_to_color/image_raw
/cam_head/d435/color/camera_info
```

## 输出字段

- `status`: 当前判定原因。
- `candidate`: 当前这一帧是否进入到达半径。
- `arrived`: 是否已连续通过并锁存到达。
- `distance_m`: 应用 `--distance-scale` 后的相机相对距离，单位为米。
- `matches`: SIFT 比率测试后的匹配数。
- `depth_matches`: 具有有效深度的匹配数。
- `inliers`: PnP RANSAC 内点数。
- `streak`: 连续通过帧数。

只有一张 JPG 时，目标图必须由同一台 D435、相近高度和内参拍摄。特征不足、
深度无效或 PnP 失败都会保守地输出“未验证”，不会误报到达。
