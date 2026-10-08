# RoadNodes 独立颜色与充分覆盖

旧分支 `feat/soft-road-bg-suppression` 的代码及诊断记录保存在提交 `7cb17f0`。本实现位于新分支 `feat/road-color-coverage`。

## 要解决的问题

旧策略监督 RoadNodes 独立 alpha，但路面 RGB 来自所有类别混合。Background 可以承担路面颜色，部分路面点在训练视角靠重叠满足覆盖目标，在新轨迹下仍然漏。

此次让 RoadNodes 独立学习真实路面/标线颜色，同时约束独立覆盖率及完整场景内的路面贡献。保留原有整场景 RGB、路面 RGB/SSIM、深度及背景软抑制。没有加入新轨迹伪 RGB，没有解冻路面位置/旋转，也没有统一扩大尺度或强制点 alpha=1。

## 颜色策略

训练中单独渲染 RoadNodes，得到预乘颜色 P 和累计 alpha A，使用 `C=P/max(A,1e-4)` 作为独立路面颜色。只用有效路面像素对比真实 RGB，按有效像素数归一化，而不是对整张黑底图平均。

P/A 的分母参与梯度：单个高斯下 C 不随其 opacity 改变，避免通过把颜色调亮来补偿低覆盖。覆盖不足另由 alpha 损失处理。多个高斯下颜色损失仍可改变它们的混合权重，并非完全冻结 opacity。

独立颜色使用完整图像相同的 affine 校正，但该损失对 affine 参数 detach，防止共享曝光模型替 RoadNodes 拟合颜色。独立颜色中不混入 Sky 或 Background。原完整场景损失仍按原方式更新共享曝光。

`road_color.w=0.5`；标线额外颜色项乘数2；标线边缘项乘数0.2。边缘损失比较水平/垂直相邻像素的真实梯度，只计算双方有效且至少一方属于标线的像素对，不惩罚mask边界。

## 覆盖策略与实际贡献

独立覆盖使用单边平方损失，普通路面目标0.95、标线目标0.98，目标以上不惩罚。全路面均值之外增加标线独立均值，避免标线面积小导致监督被稀释。`road_coverage.w=0.1`、`lane_multiplier=1`。

颜色和覆盖从第0步开始1000步warmup，避免前500步只有RGB优化先压低局部opacity。每项以独立配置控制，warmup首步系数为1/1000。

仅独立渲染仍不能保证完整图像由路面承担，因此新增类别贡献渲染：保持所有点的真实opacity、深度和排序，将RoadNodes颜色通道0置1、其他点置0。得到 `RoadNodes_contribution`，即完整场景中路面实际合成权重W。对 `W/max(A_full,1e-4)` 使用下限0.9的单边平方损失，`w=0.05`，第500步启动、1000步warmup。它同时可以提高Road opacity或降低前景Background opacity，不是只约束独立Road alpha。

两项策略的有效区域都排除ego、sky、dynamic、human、vehicle mask，并向内腐蚀3像素。不要求被车辆等真实物体遮挡的像素由RoadNodes承担。路面mask不准确仍可能造成误监督。

## 标线mask来源

Qcraft增加 `load_lane_mask=true`，`lane_mask_source=auto`：

- 优先读 `<processed scene>/lane_masks/000040_2.png` 这类二值像素标签。可配置 `lane_mask_dir`；使用 `source=labels` 时缺失标签直接报错。
- 标签缺失时，根据真实RGB在road mask内提取白色/黄色高亮且高局部对比的候选，明确输出warning和候选像素统计。这不是人工标线标签，也不把路面改画成白色：颜色目标始终取真实RGB。
- 亮度阈值0.48、局部对比阈值0.06、窗口15、白色通道差上限0.22。均可配置。阴影下标线可能漏检，亮反光/纹理可能误检；有准确mask时应优先使用。
- 图像推理和投影渲染不加载mask；低分辨率训练用max pooling缩小mask，避免细标线消失。

实际检查20260528场景第40帧相机2/3/5，已从真实RGB生成候选并排除dynamic区域、腐蚀road边界。示例：[标线监督区域](media/road_color_coverage_lane_candidates.jpg)。这只验证mask管线，并非训练画质结果。

## 开关、开销与日志

默认配置为 `configs/omnire_extended_cam_lidar.yaml`，Qcraft运行标识改为 `road-color-coverage`，避免沿用旧实验名。

- 关闭新颜色项：`trainer.losses.road_color.w=0`。
- 关闭贡献约束：`trainer.losses.road_contribution.w=0`。
- 关闭覆盖项：`trainer.losses.road_coverage.w=0`。
- 关闭标线额外监督并避免加载：`data.pixel_source.load_lane_mask=false`。
- 所有项关闭时，不触发额外训练渲染。只有颜色/覆盖时共用一次Road独立渲染；贡献项额外多一次完整类别标签渲染。对比此前coverage版本，贡献项启用后总计多一次完整光栅化，GPU耗时和显存未测。

损失日志：`road_color_loss`、`lane_color_loss`、`lane_edge_loss`、`road_coverage_loss`、`lane_coverage_loss`、`road_contribution_loss`。没有有效标线时不产生lane损失。

训练指标：`road/lane_supervised_pixels`、`road/lane_alpha`、`road/lane_low_coverage_fraction`、`road/lane_color_mae`、`road/lane_contribution_share`。低覆盖阈值读取覆盖配置。指标只在有对应输出时记录，没有mask或对应渲染时跳过。新轨迹效果仍要比较同一轨迹、同一相机、同一帧的标线和空洞，不能只看训练指标。

## 运行及验证

建议新训练而不是直接沿用此前已经饱和的opacity和尺度。旧checkpoint的模型加载兼容性未改变；从旧checkpoint继续训练时，调度使用保存的step，已经完成warmup。

现有分组入口直接使用新配置：

```bash
PYTHONPATH=. bash scripts/qcraft/group_training.sh 20260528_173056_QCJPSD850510_406_421
```

训练完成后，在输出目录对同样的新轨迹渲染，例如：

```bash
run_dir=/path/to/new/grouped/run novel_traj_shifts=left_3 \
  bash sim_render/cam/render_novel_trajectory.sh
```

验证包括12项CPU测试：颜色/opacity梯度分工、有效mask及腐蚀、标线独立归一化、边缘损失、贡献约束梯度、实际forward监督分支、实际render_fn的类别标签和info保持、affine梯度隔离、实际trainer损失入口、候选mask和标签优先级。实际代码中的CUDA依赖方法通过AST提取后配合可微合成桩测试；不是CUDA光栅化验证。原有路面软抑制3项函数测试也通过，配置按真实Qcraft合并路径检查。

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python tests/test_road_surface_prior.py
```

当前环境 `torch.cuda.is_available()` 为False，未执行GPU训练、完整CUDA反向或新轨迹画质对比。监督仍来自真实训练视角，新视角没有覆盖点或出现投影错误时不会被这个实现保证修复。新策略能减少颜色/覆盖的分工歧义，但新轨迹标线收益需实际训练验证。
