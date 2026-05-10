# 筒体论文图绘制脚本：按 PNG 拆分版

## 怎么运行

请把本文件夹中的所有 `.py` 文件放到和 `cylinder.csv` 相同的目录。

说明：

- 01-09 主图会读取 `cylinder.csv`；
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
- `plot_01_*.py` 到 `plot_09_*.py`：每个脚本只负责 1 张 PNG，便于后续单独修改。

## PNG 与代码对应关系

| 序号 | 代码文件 | 输出 PNG | 功能 |
|---:|---|---|---|
| 01-1 | `plot_01_optimization_summary.py` | `figures/cylinder/01_优化迭代过程.png` | 迭代过程图：展示可行/不可行方案与当前最优可行解。 |
| 01-2 | `plot_01_optimization_summary.py` | `figures/cylinder/02_质量-约束权衡.png` | 质量-约束权衡图：展示质量与屈曲压力之间的关系。 |
| 01-3 | `plot_01_optimization_summary.py` | `figures/cylinder/03_设计空间可行域.png` | 设计空间可行域图：展示 p-n 组合与物理约束边界。 |
| 01-4 | `plot_01_optimization_summary.py` | `figures/cylinder/04_各阶段最优已验证方案.png` | 各阶段最优方案图：展示基准、初始 LHS 和 GPR 引导验证结果。 |
| 02 | `plot_02_before_after_comparison.py` | `figures/cylinder/筒体优化前后对比.png` | 优化前后对比柱状图：展示质量下降和屈曲压力提升。 |
| 03 | `plot_03_mass_constraint_map.py` | `figures/cylinder/cylinder_mass_constraint_map.png` | 质量-屈曲压力平面图：展示可行域、不可行点、基准和最优解。 |
| 04 | `plot_04_parameter_response_grid.py` | `figures/cylinder/参数响应关系.png` | 参数-响应关系网格图：展示 t/n/p/a 与质量、屈曲压力的二维关联。 |
| 05 | `plot_05_parallel_coordinates.py` | `figures/cylinder/参数平行优化变化.png` | 平行坐标图：展示高维参数组合、响应指标和最优解轨迹。 |
| 06 | `plot_06_parameter_evolution_heatmap.py` | `figures/cylinder/圆筒参数演化热图.png` | 演化热图：观察参数与响应随优化迭代过程的变化。 |
| 08 | `plot_08_gpr_predicted_vs_actual.py` | `figures/cylinder/高斯过程回归模型验证：预测值与真实值.png` | GPR 预测值-真实值验证图：展示五折交叉验证误差和置信区间。 |
| 09 | `plot_09_gpr_response_surface.py` | `figures/cylinder/高斯过程响应面.png` | GPR 二维响应面图：固定其余参数为最优解，展示质量和屈曲压力的代理模型等高线。 |

## 01 图拆分说明

原来的 `plot_01_optimization_summary.py` 里的综合总览图已拆成：

1. `01_优化迭代过程.png`：迭代过程图；
2. `02_质量-约束权衡.png`：质量-约束权衡图；
3. `03_设计空间可行域.png`：设计空间可行域图；
4. `04_各阶段最优已验证方案.png`：各阶段最优方案图。

后续想单独改其中一张图，只需要打开对应的 `plot_01_optimization_summary.py`。
