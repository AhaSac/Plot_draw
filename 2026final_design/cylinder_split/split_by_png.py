"""
cylinder_figures_split_by_png.py

一键批量生成所有 PNG。

本版已经把“每张图的代码”拆成独立 plot_*.py 文件；
本文件只负责统一调用，方便你一次性重生成保留的图片。

输出 PNG 共 11 张：
- figures/cylinder/01_优化迭代过程.png
- figures/cylinder/02_质量-约束权衡.png
- figures/cylinder/03_设计空间可行域.png
- figures/cylinder/04_各阶段最优已验证方案.png
- figures/cylinder/筒体优化前后对比.png
- figures/cylinder/cylinder_mass_constraint_map.png
- figures/cylinder/参数响应关系.png
- figures/cylinder/参数平行优化变化.png
- figures/cylinder/圆筒参数演化热图.png
- figures/cylinder/高斯过程回归模型验证：预测值与真实值.png
- figures/cylinder/高斯过程响应面.png
"""

from cylinder_plot_common import *
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

    # GPR 相关图片共用这些中间结果，放在批量脚本里统一计算，避免重复耗时。
    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    validation, metrics = cross_validated_gpr_predictions(design, RESPONSE_COLS)
    sensitivity = gp_permutation_sensitivity(design, gpr_models)

    # 01-09：保留的论文主图；其中第 01 组现在拆成 4 张独立 PNG。
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
