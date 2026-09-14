# 路面区域 Background 软抑制方案

## 目标

本分支只抑制环境高斯对路面的侵占，不包含标线点云保留、RoadNodes
局部增密或标线像素监督。

核心原则是“抑制表达而非按路面直接删除”：高置信路面薄层内的 Background
高斯不再参与增密，并通过弱透明度正则逐渐降低贡献。该机制不直接移除高斯，
坐标也不会被正则推动。

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

## 验证建议

1. 比较路面区域 Background 的可见数量和平均透明度，不只看总高斯数。
2. 检查路缘、井盖、低矮障碍物是否仍由正确类别表达，确认没有误伤。
3. 对比 `Background_road_intrusion_opacity`、整体 PSNR 和 RoadNodes 渲染结果。
4. 记录单步耗时；正则每 4 步执行一次，增密门控只在 refinement 时执行。

## 已知边界

- 先验精度受 RoadNodes 初始化质量约束；错误的路面高度会直接影响置信度。
- 当前按中心点判断薄层，不考虑高斯椭球与路面的精确相交。
- 该分支不会立即清除已有侵占，只会让其贡献逐步降低。
