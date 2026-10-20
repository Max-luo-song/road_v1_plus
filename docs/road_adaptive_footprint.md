# 固定位置的路面高频表达改进

当前默认配置已进一步启用[多视角高频引导的局部初始化加密](road_high_frequency_init.md)，实验名为 `road-hf-init`。下文记录原 `road-adaptive-footprint` 尺度/opacity 放宽版本；关闭 `model.RoadNodes.init.high_frequency.enabled` 可只使用该基础版本。

本版基于 `feat/road-color-coverage`，位于分支 `feat/road-adaptive-footprint`。
目标是让已有 RoadNodes 在颜色交界处能够收缩或降低贡献，不依赖新增标线几何标签。
这是表达能力调整，不保证黑斑均来自尺度问题，也不承诺消除所有新轨迹伪影。

## 改动

| 属性 | 原版本 | 本版本 |
| --- | --- | --- |
| XY 标准差范围 | 2–8 cm，exp 后硬 clamp | 1–8 cm，log 空间 sigmoid 有界映射 |
| 每点 opacity 下界 | 0.2 | 0.01，保留可微下界参数化 |
| 初始化 | XY 4 cm、厚度 5 mm、opacity 0.7 | 保持同样的实际初始值 |
| 位置、旋转、厚度训练 | 冻结 | 继续冻结 |
| 路面增密 | 关闭 | 继续关闭，显式设置 edge_split_start/step=null |

对于每个高斯的两个平面尺度，使用独立参数 `u`：

```
sigma = exp(log(sigma_min) + sigmoid(u) * log(sigma_max / sigma_min))
```

这允许两个方向独立调整长宽比例；旋转仍冻结，不能保证窄轴与任意方向的标线对齐。
没有加入统一收缩正则，也不按标线 mask 强行缩小，尺度由原有图像、边缘和覆盖损失共同优化。
边界附近不合适的点可以降低 opacity；覆盖仍是累计像素覆盖，由原来的 road_coverage 损失约束。
1 cm 和 0.01 是固定的安全边界，训练自适应的是每个点的实际尺度与 opacity，并非训练边界值本身。
平滑映射消除了越过硬 clamp 后的零梯度区，但接近极限时 sigmoid 仍可能饱和。

本版不修改颜色/边缘/覆盖/贡献损失权重，不解冻旋转、不移动中心、不新增点。
原有 RGB 标线候选仍用于已有损失，不需要新增标签或 BEVRG 初始化。
尺度最大值仍为 8 cm，防止为了覆盖无界扩大；厚度范围仍为 3–10 mm。

## 配置与运行

`configs/omnire_extended_cam_lidar.yaml`：

```yaml
model:
  RoadNodes:
    ctrl:
      smooth_xy_scale: true
      scale_xy_min: 0.01
      scale_xy_max: 0.08
      edge_split_start: null
      edge_split_step: null
    reg:
      lowest_opacity: 0.01
      alpha_min: 0.01
      legacy_opacity_floor: 0.2
```

其他未启用 `smooth_xy_scale` 的配置保持原有尺度行为。该开关仅支持普通 3D RoadNodes。
实验名改为 `road-adaptive-footprint`，在本 worktree 根目录运行原分组训练入口：

```bash
PYTHONPATH=. bash scripts/qcraft/group_training.sh 20260613_102138_QCJPTDJ07384_3298_3308
```

训练入口参照 `scene_reconstruction_soft_sdf`，默认只训练、评估 group0（相机 2/3/5/9/12），不要求 group1 checkpoint；视频仅收集 group0 相机，指标直接复制 group0 结果。运行名默认仅包含 group0 相机。
GPU 沿用本目录 `scripts/qcraft/common.sh` 的 `group0_gpu` 设置，可通过 `GROUP0_GPU=0 bash scripts/qcraft/group_training.sh <scene_id>` 覆盖。启动时先检查选定 GPU 是否可用，再处理输出目录及编译 gsplat；未检测到 CUDA 时立即报错。若显式传入第二参数 `all`，仍可运行原两组流程。

新轨迹渲染默认同样只使用 group0，保留原轨迹及左移 3 m，合成时未选择的相机槽位补黑：

```bash
run_dir=/path/to/run bash sim_render/cam/render_novel_trajectory.sh
```

不再要求 group1 目录存在；需要两组时可显式设置 `render_groups=group0,group1`。
总入口 `scripts/qcraft/pipeline.sh` 同样对齐参考项目：默认使用已有 processed 数据（mode=2），仅训练 group0；可选 DiFix 流程也只处理 group0。
运行完整训练仍需原环境中的 CUDA 和第三方依赖，本次未启动训练。

## checkpoint

- 新增 `_scale_xy_bounds` buffer 标记 XY 参数含义；XY 的 `_scales` 存储有界映射的 logits，Z 仍为 log 尺度。
- 当前代码能读取原 checkpoint，先按旧版本实际渲染尺度（含原硬 clamp）解码，再转换为新参数；不会加载后统一缩小高斯。
- 原 `_opacity_alpha_min` 标记用于保留旧 opacity；无标记历史 checkpoint 按 `legacy_opacity_floor=0.2` 恢复旧下界。
- 同参数化恢复不重编码；改变尺度范围时尽可能保留实际尺寸，范围外的值会落到新边界附近。
- 导入旧模型后使用新 optimizer；原训练器仅支持模型恢复，不复用旧参数化的 optimizer 动量。
- 旧版代码不能直接读取本版 checkpoint。CPU 路面诊断脚本已更新以识别新尺度编码；其他外部导出工具不能继续把 XY `_scales` 简单取 exp。

## 验证与边界

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python tests/test_road_surface_prior.py
```

新增 CPU 测试通过 AST 提取真实 RoadNodes 类及训练器冻结钩子，避免导入 CUDA 专用依赖。
覆盖初始化实际值、两种旧 checkpoint、同参数化恢复、范围变更、参数优化越过旧下限，以及位置/旋转/厚度冻结和点数保持。
这些检查不等于 CUDA 光栅化或新轨迹画质验证。缩小尺度与降低 opacity 都可能降低覆盖，已有覆盖损失只能约束观测视角，不能保证新轨迹无洞。
