"""
cylinder_figures_split_by_png.py

一键批量生成所有 PNG。

本版已经把“每张图的代码”拆成独立 plot_*.py 文件；
本文件只负责统一调用，方便你一次性重生成全部图片和 CSV 指标表。

输出 PNG 共 14 张：
- figures/cylinder/cylinder_optimization_summary.png
- figures/cylinder/cylinder_before_after_comparison.png
- figures/cylinder/cylinder_mass_constraint_map.png
- figures/cylinder/cylinder_parameter_response_grid.png
- figures/cylinder/cylinder_parallel_coordinates.png
- figures/cylinder/cylinder_parameter_evolution_heatmap.png
- figures/cylinder/cylinder_sampling_scatter_matrix.png
- figures/cylinder/cylinder_gpr_predicted_vs_actual.png
- figures/cylinder/cylinder_gpr_response_surface.png
- figures/cylinder/cylinder_gpr_sensitivity_bar.png
- figures/cylinder/cylinder_gpr_sobol_bar.png
- figures/cylinder/cylinder_viz_mass_margin_pareto.png
- figures/cylinder/cylinder_viz_parallel_coordinates.png
- figures/cylinder/cylinder_viz_mass_correlation.png
"""

from cylinder_plot_common import *
from plot_01_optimization_summary import draw_summary_figure
from plot_02_before_after_comparison import draw_before_after_comparison
from plot_03_mass_constraint_map import draw_mass_constraint_map
from plot_04_parameter_response_grid import draw_parameter_response_grid
from plot_05_parallel_coordinates import draw_parallel_coordinates
from plot_06_parameter_evolution_heatmap import draw_parameter_evolution_heatmap
from plot_07_sampling_scatter_matrix import draw_sampling_scatter_matrix
from plot_08_gpr_predicted_vs_actual import draw_gpr_validation
from plot_09_gpr_response_surface import draw_gpr_response_surface
from plot_10_gpr_sensitivity_bar import draw_gpr_sensitivity_bar
from plot_11_gpr_sobol_bar import draw_gpr_sobol_bar
from plot_12_viz_mass_margin_pareto import draw_viz_mass_margin_pareto
from plot_13_viz_parallel_coordinates import draw_viz_parallel_coordinates
from plot_14_viz_mass_correlation import draw_viz_mass_correlation


def main() -> None:
    """批量运行所有绘图函数，并导出复核用 CSV 表。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()

    # GPR 相关图片共用这些中间结果，放在批量脚本里统一计算，避免重复耗时。
    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    validation, metrics = cross_validated_gpr_predictions(design, RESPONSE_COLS)
    sensitivity = gp_permutation_sensitivity(design, gpr_models)
    sobol_df = sobol_sensitivity_from_gp(design, gpr_models)

    # 01-11：论文主图，每个函数对应一个 PNG 文件。
    draw_summary_figure(baseline, design, best, pressure_limit)
    draw_before_after_comparison(baseline, best)
    draw_mass_constraint_map(baseline, design, best, pressure_limit)
    draw_parameter_response_grid(design, best, pressure_limit)
    draw_parallel_coordinates(design, best)
    draw_parameter_evolution_heatmap(design, best)
    draw_sampling_scatter_matrix(design, best)
    draw_gpr_validation(validation, metrics)
    draw_gpr_response_surface(design, best, gpr_models, sensitivity, pressure_limit)
    draw_gpr_sensitivity_bar(sensitivity)
    draw_gpr_sobol_bar(sobol_df)

    # 12-14：原 viz 三联图已拆成三张独立 PNG。
    draw_viz_mass_margin_pareto()
    draw_viz_parallel_coordinates()
    draw_viz_mass_correlation()

    # CSV 指标表：用于论文复核，不属于图片。
    ranked_path = export_ranked_table(design)
    metrics_path = OUT_DIR / "cylinder_gpr_validation_metrics.csv"
    validation_path = OUT_DIR / "cylinder_gpr_validation_predictions.csv"
    sensitivity_path = OUT_DIR / "cylinder_gpr_sensitivity.csv"
    sobol_path = OUT_DIR / "cylinder_gpr_sobol_indices.csv"
    metrics.to_csv(metrics_path, index=False)
    validation.to_csv(validation_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    sobol_df.to_csv(sobol_path, index=False)
    cleanup_temp_files()

    print(f"Style: {style_name}")
    print(f"Rows: {len(design)} verified designs")
    print(f"Feasible designs: {int(design['feasible'].sum())}")
    print(f"Current optimum: {best['case_name']}")
    print(f"Saved figures to: {OUT_DIR}")
    print(f"Saved ranked feasible table to: {ranked_path}")
    print(f"Saved GPR validation metrics to: {metrics_path}")
    print(f"Saved GPR validation predictions to: {validation_path}")
    print(f"Saved GPR sensitivity table to: {sensitivity_path}")
    print(f"Saved Sobol sensitivity table to: {sobol_path}")


if __name__ == "__main__":
    main()
