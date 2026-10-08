# 路面不透明度优化增量接入

参考 `scene_reconstruction_lane_opacity`，本次接入路面覆盖率监督及可微不透明度下限。当前的 Background 路面侵占抑制、路面/非路面 RGB 与 SSIM 损失、标线几何初始化均保留。

## 新增策略

- 训练中额外渲染 RoadNodes，输出 `RoadNodes_opacity`。只在配置权重大于 0、超过启动步数且存在 road mask 时执行。
- 覆盖率损失为有效像素内 `relu(target_alpha - opacity)^2` 的均值。目标以上不惩罚。
- 有效区域为路面 mask，排除 ego 和 dynamic 区域，再向内腐蚀 3 像素。
- 默认目标为 0.95；若输入提供 `lane_masks`，标线目标为 0.98。当前数据加载流程没有像素级 `lane_masks`，因此默认全部使用 0.95。本次没有引入参考项目的标线增密和独立标线 RGB 监督。
- 第 500 步之后启动，1000 步线性 warmup，最终权重 0.1。日志新增 `road_coverage_loss`。
- 显式配置 `alpha_min` 时使用 `alpha_min + (1-alpha_min) * sigmoid(raw)`，默认下限 0.2、初始实际不透明度 0.7。alpha reset 使用对应逆变换，稀疏正则使用实际不透明度。

## 配置与兼容

`configs/omnire_extended_cam_lidar.yaml` 已启用新策略。其他配置未配置 `alpha_min` 时保留原 sigmoid + clamp 路径；未配置 `road_coverage` 时不增加覆盖率监督或额外渲染。

保留原有 `lowest_opacity: 0.2`。新策略启用时由 `alpha_min` 控制下限。关闭新策略时删除 `alpha_min`，并将 `road_coverage.w` 设为 0；如需恢复原初始化行为，同时删除 `opacity_init_value`。

checkpoint 新增 `_opacity_alpha_min` buffer，标记保存时使用的参数化。旧 checkpoint 缺少该字段时按普通 sigmoid 解释，并在需要时转换为当前参数化；同参数化的 checkpoint 直接恢复。转换时下限位置受逆变换数值截断影响，有微小误差。参数化改变时历史 optimizer 动量不会进行坐标变换，建议仅恢复模型后重新创建 optimizer。

## 验证与开销

已检查修改文件的 Python 语法，并验证可微下限梯度、正逆变换、mask 腐蚀、覆盖率损失的梯度方向及目标以上零惩罚。未执行完整 GPU 训练，实际画质收益需要训练对比。

启用覆盖率监督后每个训练步多一次 RoadNodes 渲染，会增加耗时及显存占用。该损失经过可微渲染，也可能更新 RoadNodes 几何参数。
