"""
head_figures_split_by_png.py

一键批量生成所有 PNG。

本版已经把“每张图的代码”拆成独立 plot_*.py 文件；
本文件只负责统一调用，方便你一次性重生成保留的图片。

输出 PNG 共 11 张：
- figures/head/01_优化迭代过程.png
- figures/head/02_质量-约束权衡.png
- figures/head/03_设计空间可行域.png
- figures/head/04_各阶段最优已验证方案.png
- figures/head/封头优化前后对比.png
- figures/head/封头质量-约束权衡.png
- figures/head/参数响应关系.png
- figures/head/参数平行优化变化.png
- figures/head/封头参数演化热图.png
- figures/head/高斯过程回归模型验证：预测值与真实值.png
- figures/head/高斯过程响应面.png
"""

from head_plot_common import *
from plot_01_optimization_summary import draw_summary_figure
from plot_02_before_after_comparison import draw_before_after_comparison
from plot_03_mass_constraint_map import draw_mass_constraint_map
from plot_04_parameter_response_grid import draw_parameter_response_grid
from plot_05_parallel_coordinates import draw_parallel_coordinates
from plot_06_parameter_evolution_heatmap import draw_parameter_evolution_heatmap
from plot_08_gpr_predicted_vs_actual import draw_gpr_validation
from plot_09_gpr_response_surface import draw_gpr_response_surface


def main() -> None:
    """批量运行保留的绘图函数。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()

    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    validation, metrics = cross_validated_gpr_predictions(design, RESPONSE_COLS)
    sensitivity = gp_permutation_sensitivity(design, gpr_models)

    draw_summary_figure(baseline, design, best, pressure_limit)
    draw_before_after_comparison(baseline, best)
    draw_mass_constraint_map(baseline, design, best, pressure_limit)
    draw_parameter_response_grid(design, best, pressure_limit)
    draw_parallel_coordinates(design, best)
    draw_parameter_evolution_heatmap(design, best)
    draw_gpr_validation(validation, metrics)
    draw_gpr_response_surface(design, best, gpr_models, sensitivity, pressure_limit)
    cleanup_temp_files()

    print(f"Style: {style_name}")
    print(f"Rows: {len(design)} verified designs")
    print(f"Feasible designs: {int(design['feasible'].sum())}")
    print(f"Current optimum: {best['case_name']}")
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
