# 环境抑制相关代码改动说明

本文档基于当前分支 `feat/soft-road-bg-suppression` 的代码和提交历史整理，目标是完整说明“Background 环境高斯侵占路面”的软抑制实现、配置、测试和当前工作区状态。

## 1. 范围与提交时间线

当前分支相对项目快照 `297a918` 的主要主题改动来自：

| 提交 | 内容 | 与环境抑制的关系 |
| --- | --- | --- |
| `3639b1b` | `feat: softly suppress background road intrusion` | 核心算法实现 |
| `1dde071` | `chore: save soft road background suppression state` | 保存实验运行状态；包含实验标识及若干脚本/配置变化 |

此前的 `5451bcb` 是车道线密集监督和 RoadNodes 覆盖率监督，提交文档明确说明“不包含 Background 路面侵占抑制逻辑”，因此不计入本机制的核心实现。不过，本机制复用了它产生的 `RoadNodes` 路面点作为高度先验。

相对 `297a918`，环境抑制主题涉及的文本/代码文件为：

* `configs/omnire_extended_cam_lidar.yaml`
* `models/gaussians/vanilla.py`
* `models/trainers/scene_graph.py`
* `utils/road_surface_prior.py`
* `tests/test_road_surface_prior.py`
* `docs/soft_road_background_suppression.md`
* `scripts/qcraft/common.sh`（实验状态和当前未提交运行参数）
* `sim_render/cam/render_novel_trajectory.sh`（同一保存状态提交中的默认评估轨迹）

同一提交还加入了 `.sii/ide-port` 和两个输出媒体文件，它们不改变训练算法。

## 2. 设计目标和明确边界

实现针对的是 **Background 类高斯对路面薄层的侵占**。核心策略是“抑制表达，不按路面直接删除”：

1. 路面由 `RoadNodes` 的中心点构造高度先验。
2. Background 高斯中心落在路面支持区域且位于路面高度附近时，得到一个 0 到 1 的连续侵占置信度。
3. 高置信 Background 高斯不再参与后续 split/duplicate 增密。
4. 对可见高斯增加弱透明度正则，让其贡献逐步降低。

该机制不做以下事情：

* 不直接删除或 prune 路面侵占高斯；
* 不移动高斯中心，不对位置施加路面法向/高度正则；
* 不包含标线点云保留、RoadNodes 局部增密或标线像素监督；
* 不替换已有的通用低透明度和过大高斯清理逻辑。

因此，侵占高斯可能在训练中继续存在，只是透明度逐渐降低；当透明度满足既有通用清理条件时，才可能被原逻辑移除。

## 3. 配置改动

文件：`configs/omnire_extended_cam_lidar.yaml`，位置为 `model.Background.reg`。

```yaml
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

参数含义：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `opacity_w` | `0.005` | 透明度正则权重。值越大，侵占高斯透明度下降越快，但可能影响路面附近真实物体表达。 |
| `start_after` | `500` | 训练步数达到该值后才启用增密门控和透明度正则。 |
| `step_interval` | `4` | 每隔 4 步计算一次透明度正则，降低额外开销。 |
| `grid_size` | `0.25 m` | XY 高度网格初始分辨率。 |
| `max_grid_cells` | `4,000,000` | 高度网格的最大单元数，防止大场景导致内存无界增长。 |
| `below_margin` | `0.1 m` | 高斯中心允许低于路面高度的范围。 |
| `above_margin` | `0.15 m` | 高斯中心允许高于路面高度的范围。 |
| `transition` | `0.05 m` | 上下边界的 sigmoid 软过渡宽度。 |
| `densify_confidence` | `0.5` | 侵占置信度达到该值时，禁止该高斯触发 split/duplicate。 |

## 4. 路面高度先验：`utils/road_surface_prior.py`

### 4.1 `build_road_surface_grid`

函数输入为非空 `[N, 3]` 的 RoadNodes 点坐标，输出：

* `grid_origin`：XY 最小值，作为网格原点；
* `grid_heights[height, width]`：每个格子的路面 Z 均值；
* `grid_valid[height, width]`：该格子是否有 RoadNodes 点支撑；
* `effective_cell_size`：实际使用的网格边长。

实现步骤：

1. 用 RoadNodes 点的 XY 最小/最大值计算场景跨度。
2. 从配置的 `cell_size` 开始计算网格尺寸，若单元总数超过 `max_grid_cells`，按跨度比例增大网格边长并重新计算。
3. 将每个点映射到 XY 格子，使用 `scatter_add_` 累加 Z 和点数。
4. 用 `height_sums / counts` 得到每格平均路面高度，`counts > 0` 作为有效性掩码。

函数带 `@torch.no_grad()`，先验构建不进入反向图。网格只记录有 RoadNodes 支撑的格子，网格外或无支撑格子不会被认为是路面。

### 4.2 `road_intrusion_confidence`

对一批 Background 中心点计算置信度：

1. 用 `(x, y)` 和网格原点/边长得到格子索引。
2. 检查索引是否在网格内、格子是否有效；不满足时置信度为 0。
3. 计算 `height_delta = gaussian_z - road_height`。
4. 分别计算上下边界的 sigmoid：

```text
lower = sigmoid((height_delta + below_margin) / transition)
upper = sigmoid((above_margin - height_delta) / transition)
confidence = lower * upper * supported
```

因此，在路面下约 10 cm 到路面上约 15 cm 的中心点置信度最高，边界宽度约 5 cm，避免硬阈值造成训练突变。这里判断的是高斯**中心点**，没有计算高斯椭球与路面薄层的精确相交。

## 5. Background 高斯类改动：`models/gaussians/vanilla.py`

### 5.1 状态初始化

`VanillaGaussians.__init__` 新增三个非持久 buffer：

* `_road_grid_origin`
* `_road_grid_heights`
* `_road_grid_valid`

另有普通成员 `_road_grid_cell_size`。它们使用 `persistent=False`，不会写入 checkpoint。这样 checkpoint 只保存模型参数，恢复后由当前 RoadNodes 重新构建先验。

### 5.2 `set_road_surface_prior`

该方法读取 `self.reg_cfg["road_intrusion"]`。没有配置或没有 RoadNodes 点时直接返回；否则调用 `build_road_surface_grid`，并记录网格尺寸和实际边长日志。

### 5.3 `get_road_intrusion_confidence`

该方法是 Background 模型访问置信度的统一入口：没有配置或先验尚未建立时返回全 0，正常情况下将配置的上下边界和过渡宽度传给 `road_intrusion_confidence`。

### 5.4 增密门控

位置：`refinement_after` 的 split/duplicate 选择阶段。

原逻辑先根据累计 XY 梯度得到 `high_grads`。当满足以下条件时增加门控：

* 已配置 `road_intrusion`；
* 路面网格已建立；
* 当前步 `step >= start_after`。

此时计算所有 Background 高斯的侵占置信度，并执行：

```python
high_grads &= intrusion < densify_confidence
```

后续的 `splits` 和 `dups` 都依赖这个 `high_grads`，所以高置信侵占点不会继续分裂或复制；低置信点仍按原有尺度、屏幕尺寸和梯度规则增密。该门控只影响增密，不是删除条件。

### 5.5 透明度正则

位置：`compute_reg_loss`。

当配置存在、先验已建立、步数达到 `start_after` 且步数满足 `step_interval` 时：

1. 只选择 `self.cur_radii > 0` 的当前可见高斯；
2. 计算可见高斯的侵占置信度，并立即 `.detach()`；
3. 读取 `torch.sigmoid(self._opacities)` 作为透明度；
4. 计算置信度加权平均透明度，并乘以 `opacity_w`，写入 `loss_dict["road_intrusion_opacity"]`。

等价形式为：

```text
loss = opacity_w * sum(opacity_i * confidence_i) / max(sum(confidence_i), 1e-6)
```

由于置信度与位置计算结果被分离，反向传播只会通过 opacity 参数，理论上不会通过该项移动高斯坐标。若当前没有可见高斯，则不添加该 loss 项。

## 6. 训练器接入：`models/trainers/scene_graph.py`

新增 `_configure_road_background_prior`：

* 只有同时存在 `Background` 和 `RoadNodes` 时才执行；
* 通过 `Background.set_road_surface_prior(RoadNodes._means)` 建立路面网格。

该方法在两条生命周期路径调用：

1. `init_gaussians_from_dataset` 完成模型初始化、删除空类别后调用；
2. `load_state_dict` 恢复 checkpoint 后调用。

这保证了新训练和断点恢复都能建立先验，也解释了为什么网格不需要保存到 checkpoint。先验来源是 RoadNodes 当前中心坐标，因此 RoadNodes 初始化质量会直接影响抑制范围。

## 7. 测试覆盖：`tests/test_road_surface_prior.py`

新增 3 个测试：

1. `test_road_intrusion_confidence_is_local_and_height_aware`：验证同一有效格子、同一路面高度附近置信度较高；高于路面 1 m 或位于网格外时置信度接近/等于 0。
2. `test_road_surface_grid_respects_memory_bound`：验证大跨度点集会自动放大有效网格边长，并确保网格元素数不超过 `max_grid_cells`。
3. `test_detached_confidence_only_updates_opacity`：验证置信度 detach 后，透明度 logit 能得到梯度，而高斯中心 `means` 不得到梯度。

这些测试覆盖了核心数学函数和 detach 语义，没有覆盖完整训练循环、实际渲染结果、checkpoint 恢复后的网格重建或性能基准。

## 8. 实验运行状态相关改动

### 8.1 `scripts/qcraft/common.sh`

在提交 `1dde071` 中，实验输出标识从可由环境变量覆盖改为固定值：

```bash
extra_config_info="soft-road-bg-suppression"
```

它用于输出目录/实验名称区分，帮助把结果标记为本实验；不参与模型计算。

当前工作区还有一项**未提交**变化：

```diff
-group0_gpu=2
-group1_gpu=2
+group0_gpu=0
+group1_gpu=0
```

这只改变相机分组训练使用的 GPU，不改变环境抑制算法。当前工作区另有未跟踪文件 `debug_model_nan_1.pth`，它不是环境抑制代码。

### 8.2 `sim_render/cam/render_novel_trajectory.sh`

同一保存状态提交把默认 `novel_traj_shifts` 从六个左右轨迹改为仅 `left_3`。这是评估/渲染范围变化，不影响训练中的路面抑制逻辑。

### 8.3 配置中移除 `RoadNodes.ctrl.lane_influence_radius`

`1dde071` 从 YAML 删除了 `lane_influence_radius: 0.2`。该参数属于 RoadNodes 车道线影响范围逻辑，不是 Background 路面侵占抑制参数；代码仍通过 `get("lane_influence_radius", 0.2)` 提供默认值。因此这项删除不会关闭本环境抑制机制，但会让 RoadNodes 使用代码默认值。

## 9. 训练时序总结

```text
初始化/恢复 RoadNodes
        |
        v
用 RoadNodes._means 建立 Background 路面高度网格
        |
        v
前 500 步：不启用路面侵占门控和正则
        |
        +--> refinement：高置信侵占点禁止 split/duplicate
        |
        +--> 每 4 步：对可见侵占点增加透明度正则
        |
        v
透明度逐步降低；后续是否删除仍由既有通用清理规则决定
```

## 10. 已知限制与排查重点

* RoadNodes 高度先验错误会把错误区域标为路面，造成误抑制；应检查 RoadNodes 初始化质量。
* 网格按 XY 格子平均 Z，坡道、台阶和多层结构在同一格子内可能被平均化。
* 只按中心点判断，不考虑高斯尺度；大高斯可能部分穿过路面但不被识别，或中心命中而实际覆盖了路面以上物体。
* 透明度正则只处理当前可见点，遮挡或当前视角不可见的侵占点不会在该步获得抑制梯度。
* 正则是弱约束且默认权重很小，不保证短时间内清除已有侵占。
* 正则 loss 使用 `torch.sigmoid(self._opacities)`，与 Background 当前 opacity 激活方式一致时行为正常；若模型子类改变 opacity 参数化，需要同步检查。
* 建议同时记录 `Background_road_intrusion_opacity`、路面区域可见高斯数量、平均透明度、整体 PSNR 和 RoadNodes 渲染结果，而不是只看总高斯数。

