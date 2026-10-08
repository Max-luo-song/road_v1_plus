# 路面区域 Background 软抑制方案

本文说明当前代码中“环境抑制高斯”的完整实现。这里的“环境高斯”指
`Background` 类高斯；抑制目标是它们在路面薄层中的错误表达（例如路面上方的
Background 高斯侵占路面），不是 `RoadNodes` 本身。

## 一、实现总览

实现可以概括为以下数据流：

```text
RoadNodes._means
    -> build_road_surface_grid()
    -> Background 路面高度先验
    -> road_intrusion_confidence(Background._means)
        -> refinement_after(): 禁止高置信点 split/duplicate
        -> compute_reg_loss(): 对可见高置信点施加透明度正则
```

对应代码入口如下：

| 环节 | 代码 | 作用 |
| --- | --- | --- |
| 高度先验构建 | `utils/road_surface_prior.py::build_road_surface_grid` | 将 RoadNodes 点按 XY 网格聚合为路面高度和有效区域 |
| 侵占判定 | `utils/road_surface_prior.py::road_intrusion_confidence` | 根据 XY 网格支持关系和 Z 高度差返回 `[0, 1]` 置信度 |
| Background 封装 | `models/gaussians/vanilla.py::set_road_surface_prior`、`get_road_intrusion_confidence` | 保存先验并提供统一查询接口 |
| 增密抑制 | `models/gaussians/vanilla.py::refinement_after` | 在 split/duplicate 前过滤高置信侵占点 |
| 透明度抑制 | `models/gaussians/vanilla.py::compute_reg_loss` | 对当前可见点增加透明度损失 |
| 生命周期接入 | `models/trainers/scene_graph.py::_configure_road_background_prior` | 初始化和恢复 checkpoint 后重建先验 |
| 参数配置 | `configs/omnire_extended_cam_lidar.yaml::model.Background.reg.road_intrusion` | 控制权重、时机、网格和阈值 |

## 目标

本分支只抑制环境高斯对路面的侵占，不包含标线点云保留、RoadNodes
局部增密或标线像素监督。

核心原则是“抑制表达而非按路面直接删除”：高置信路面薄层内的 Background
高斯不再参与增密，并通过弱透明度正则逐渐降低贡献。该机制不直接移除高斯，
坐标也不会被正则推动。

## 二、路面高度先验

### 2.1 为什么使用 RoadNodes

`MultiTrainer._configure_road_background_prior()` 只有在同时存在
`Background` 和 `RoadNodes` 时才执行，并把 `RoadNodes._means` 传入
`Background.set_road_surface_prior()`。该方法在
`init_gaussians_from_dataset()` 完成模型初始化后调用一次，在
`load_state_dict()` 恢复模型后再次调用。因此网格是由当前 RoadNodes 重建的，
不依赖 checkpoint 保存网格缓存。

### 2.2 网格聚合

`build_road_surface_grid()`（`utils/road_surface_prior.py`）执行：

1. 用 RoadNodes 的 XY 最小值作为原点，用 XY 跨度和 `grid_size` 计算网格大小。
2. 若网格单元总数超过 `max_grid_cells`，自动增大实际格距，限制显存和内存开销。
3. 将每个点映射到格子，用 `scatter_add_` 累加 Z 和点数。
4. 输出每格平均高度 `grid_heights` 和有点支撑的掩码 `grid_valid`。

网格外或没有 RoadNodes 支撑的格子不参与抑制，这使抑制范围局部依赖于路面先验，
不会把整个场景的 Background 都当成路面。

## 三、连续侵占置信度

`road_intrusion_confidence()` 只检查高斯中心，不计算高斯椭球和路面的精确相交。
对每个 Background 中心 `(x, y, z)`：

1. 由 XY 查找对应网格格子，并检查格子是否在网格内且 `grid_valid=True`。
2. 读取该格子的路面高度 `road_height`，计算 `height_delta = z - road_height`。
3. 使用上下两个 sigmoid 形成软薄层：

```text
lower = sigmoid((height_delta + below_margin) / transition)
upper = sigmoid((above_margin - height_delta) / transition)
confidence = lower * upper * supported
```

默认情况下，路面下约 10 cm 到路面上约 15 cm 的中心点置信度最高，边界以 5 cm
平滑过渡。置信度是连续值而不是硬布尔条件，避免高斯在边界处突然改变训练行为。

## 实现

### 一次性路面高度先验

- RoadNodes 初始化或 checkpoint 恢复后，将其点坐标聚合为 XY 高度网格。
- 默认网格分辨率为 25 cm，只记录有 RoadNodes 支撑的格子。
- 网格最多 400 万格；场景跨度异常大时自动增大格距，避免内存无界增长。
- 高度网格不写入 checkpoint，恢复后从 RoadNodes 自动重建。

### 连续侵占置信度

Background 中心落在有路面支撑的格子，并处于路面下 10 cm 到上 15 cm
的薄层时获得高置信度。边界使用 5 cm 软过渡，而不是硬阈值。

### 非删除式抑制

- 从第 500 步开始，高置信 Background 不再触发 split/duplicate。
- 每 4 步只对当前可见 Background 计算一次透明度正则，默认权重 `0.005`。
- 置信度计算从梯度图中分离，所以该正则只更新透明度，不修改高斯坐标。
- 不新增基于路面置信度的 prune/cull 条件；原有通用低透明度和过大高斯清理逻辑保持不变。因此，贡献长期降低到通用阈值以下的点，后续仍可能被原逻辑清理。

### 代码中的两个抑制动作

**1. 增密门控（`VanillaGaussians.refinement_after`）**

在原有平均 XY 梯度阈值生成 `high_grads` 后，代码在满足以下条件时查询侵占置信度：

- 已配置 `road_intrusion` 且路面网格已建立；
- 当前训练步 `step >= start_after`。

随后执行 `high_grads &= intrusion < densify_confidence`。由于 `splits` 和 `dups`
都依赖 `high_grads`，高置信侵占点不会继续 split/duplicate，低置信点仍使用原来的
尺度、屏幕尺寸和梯度规则增密。这一步不改变现有高斯，也不是删除条件。

**2. 透明度正则（`VanillaGaussians.compute_reg_loss`）**

满足启动步数和 `step_interval` 后，代码只取 `self.cur_radii > 0` 的当前可见高斯，
计算其侵占置信度并 `.detach()`，再对 `sigmoid(self._opacities)` 做置信度加权平均：

```text
road_intrusion_opacity
  = opacity_w * sum(opacity_i * confidence_i)
                   / max(sum(confidence_i), 1e-6)
```

`confidence.detach()` 保证该项的梯度只更新 opacity，不通过侵占判定移动高斯中心；
不可见高斯在当前步不产生该项梯度。该损失写入 `loss_dict`，由训练器与其他正则项
一起反向传播。

## 配置

唯一的 YAML 改动位于 `configs/omnire_extended_cam_lidar.yaml`：

```yaml
model:
  Background:
    reg:
      road_intrusion:
        opacity_w: 0.005
        start_after: 500
        step_interval: 4
        grid_size: 0.25
        max_grid_cells: 4_000_000
        below_margin: 0.1
        above_margin: 0.15
        transition: 0.05
        densify_confidence: 0.5
```

参数与代码使用位置对应如下：

| 参数 | 默认值 | 使用位置和含义 |
| --- | ---: | --- |
| `opacity_w` | `0.005` | `compute_reg_loss` 中透明度损失的权重 |
| `start_after` | `500` | `refinement_after` 和 `compute_reg_loss` 的共同启动步数 |
| `step_interval` | `4` | 透明度正则的计算间隔；增密门控仍只在 refinement 时触发 |
| `grid_size` | `0.25` | 高度网格的目标 XY 格距（米） |
| `max_grid_cells` | `4_000_000` | `build_road_surface_grid` 的最大单元数 |
| `below_margin` | `0.1` | 路面以下的软边界（米） |
| `above_margin` | `0.15` | 路面以上的软边界（米） |
| `transition` | `0.05` | sigmoid 过渡宽度（米） |
| `densify_confidence` | `0.5` | 置信度达到该值时禁止 split/duplicate |

未配置 `road_intrusion`、没有 RoadNodes，或先验尚未建立时，
`get_road_intrusion_confidence()` 返回全零，Background 训练退化为原有逻辑。

## 验证建议

1. 比较路面区域 Background 的可见数量和平均透明度，不只看总高斯数。
2. 检查路缘、井盖、低矮障碍物是否仍由正确类别表达，确认没有误伤。
3. 对比 `Background_road_intrusion_opacity`、整体 PSNR 和 RoadNodes 渲染结果。
4. 记录单步耗时；正则每 4 步执行一次，增密门控只在 refinement 时执行。

单元测试位于 `tests/test_road_surface_prior.py`，覆盖：局部且高度相关的置信度、
网格单元数上限，以及 `confidence.detach()` 后只有 opacity 获得梯度。完整训练循环、
实际渲染质量和 checkpoint 重建仍需通过实验验证。

## 已知边界

- 先验精度受 RoadNodes 初始化质量约束；错误的路面高度会直接影响置信度。
- 当前按中心点判断薄层，不考虑高斯椭球与路面的精确相交。
- 该分支不会立即清除已有侵占，只会让其贡献逐步降低。
