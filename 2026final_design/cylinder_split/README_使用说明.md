# 筒体论文图绘制脚本：按 PNG 拆分版

## 怎么运行

请把本文件夹中的所有 `.py` 文件放到和 `cylinder.csv` 相同的目录。

说明：

- 01-11 主图会读取 `cylinder.csv`；
- 12-14 的 viz 拆分图直接使用脚本中内置的 `VIZ_CASE_DATA`，不需要 CSV；
- 现在已经不再需要 `cylinder_gpr_optimization.py`，GPR 类已经内置到 `cylinder_plot_common.py`。

然后运行：

```bash
python cylinder_figures_split_by_png.py
```

所有 PNG 会输出到：

```text
figures/cylinder/
```

## 文件结构

- `cylinder_plot_common.py`：公共配置、数据读取、辅助函数、自包含 GPR 计算函数。
- `cylinder_figures_split_by_png.py`：一键批量生成全部图片。
- `plot_01_*.py` 到 `plot_14_*.py`：每个脚本只负责 1 张 PNG，便于后续单独修改。

## PNG 与代码对应关系

| 序号 | 代码文件 | 输出 PNG | 功能 |
|---:|---|---|---|
| 01 | `plot_01_optimization_summary.py` | `figures/cylinder/cylinder_optimization_summary.png` | 综合总览图：迭代过程、质量-约束权衡、可行域与阶段最优。 |
| 02 | `plot_02_before_after_comparison.py` | `figures/cylinder/cylinder_before_after_comparison.png` | 优化前后对比柱状图：展示质量下降和屈曲压力提升。 |
| 03 | `plot_03_mass_constraint_map.py` | `figures/cylinder/cylinder_mass_constraint_map.png` | 质量-屈曲压力平面图：展示可行域、不可行点、基准和最优解。 |
| 04 | `plot_04_parameter_response_grid.py` | `figures/cylinder/cylinder_parameter_response_grid.png` | 参数-响应关系网格图：展示 t/n/p/a 与质量、屈曲压力的二维关联。 |
| 05 | `plot_05_parallel_coordinates.py` | `figures/cylinder/cylinder_parallel_coordinates.png` | 平行坐标图：展示高维参数组合、响应指标和最优解轨迹。 |
| 06 | `plot_06_parameter_evolution_heatmap.py` | `figures/cylinder/cylinder_parameter_evolution_heatmap.png` | 演化热图：观察参数与响应随优化迭代过程的变化。 |
| 07 | `plot_07_sampling_scatter_matrix.py` | `figures/cylinder/cylinder_sampling_scatter_matrix.png` | 采样散点矩阵：对比初始 LHS、GPR 引导点和随机参考分布。 |
| 08 | `plot_08_gpr_predicted_vs_actual.py` | `figures/cylinder/cylinder_gpr_predicted_vs_actual.png` | GPR 预测值-真实值验证图：展示五折交叉验证误差和置信区间。 |
| 09 | `plot_09_gpr_response_surface.py` | `figures/cylinder/cylinder_gpr_response_surface.png` | GPR 二维响应面图：固定其余参数为最优解，展示质量和屈曲压力的代理模型等高线。 |
| 10 | `plot_10_gpr_sensitivity_bar.py` | `figures/cylinder/cylinder_gpr_sensitivity_bar.png` | GPR 置换敏感性柱状图：比较各设计参数对质量和屈曲压力的影响。 |
| 11 | `plot_11_gpr_sobol_bar.py` | `figures/cylinder/cylinder_gpr_sobol_bar.png` | Sobol 全局敏感性柱状图：并排展示一阶指数 S1 与总阶指数 ST。 |
| 12 | `plot_12_viz_mass_margin_pareto.py` | `figures/cylinder/cylinder_viz_mass_margin_pareto.png` | 原 viz 三联图第 1 张：质量-屈曲裕度权衡散点图与 Pareto 前沿。 |
| 13 | `plot_13_viz_parallel_coordinates.py` | `figures/cylinder/cylinder_viz_parallel_coordinates.png` | 原 viz 三联图第 2 张：低质量/高质量方案的平行坐标对比。 |
| 14 | `plot_14_viz_mass_correlation.py` | `figures/cylinder/cylinder_viz_mass_correlation.png` | 原 viz 三联图第 3 张：设计变量与质量的相关性条形图。 |

## viz 三联图拆分说明

原来的 `cylinder_viz_tradeoff.png` 已拆成：

1. `cylinder_viz_mass_margin_pareto.png`：质量-屈曲裕度 + Pareto 前沿；
2. `cylinder_viz_parallel_coordinates.png`：viz 数据平行坐标；
3. `cylinder_viz_mass_correlation.png`：设计变量与质量相关性。

后续想单独改其中一张图，只需要打开对应的 `plot_12_*.py`、`plot_13_*.py` 或 `plot_14_*.py`。
