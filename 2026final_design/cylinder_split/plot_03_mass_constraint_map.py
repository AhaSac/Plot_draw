"""
plot_03_mass_constraint_map.py

输出 PNG：figures/cylinder/cylinder_mass_constraint_map.png
功能：质量-屈曲压力平面图：展示可行域、不可行点、基准和最优解。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_mass_constraint_map() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_mass_constraint_map.png ==============================
# 功能：质量-屈曲压力平面图：展示可行域、不可行点、基准和最优解。
def draw_mass_constraint_map(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """质量-屈曲压力平面图：直观展示可行域和最优解位置。"""
    fig, ax = plt.subplots(figsize=(3.55, 2.65), constrained_layout=True)

    feasible = design[design["feasible"]]
    infeasible = design[~design["feasible"]]

    ax.axhspan(pressure_limit, design["buckling_pressure"].max() * 1.08, color=COLORS["pale_green"], zorder=0)
    ax.axhspan(0, pressure_limit, color=COLORS["pale_red"], zorder=0)
    ax.axhline(pressure_limit, color=COLORS["black"], lw=0.85, ls=(0, (3, 2)))
    ax.scatter(
        infeasible["total_mass"],
        infeasible["buckling_pressure"],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.82,
        label="不可行",
    )
    ax.scatter(
        feasible["total_mass"],
        feasible["buckling_pressure"],
        s=27,
        color=COLORS["green"],
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        label="可行",
    )
    ax.scatter(
        float(baseline["total_mass"]),
        pressure_limit,
        marker="s",
        s=32,
        color=COLORS["blue"],
        edgecolors=COLORS["black"],
        linewidths=0.4,
        label="基准",
        zorder=4,
    )
    ax.scatter(
        best["total_mass"],
        best["buckling_pressure"],
        marker="*",
        s=130,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        label="最优解",
        zorder=5,
    )
    annotate_best(
        ax,
        best["total_mass"],
        best["buckling_pressure"],
        f"{best['case_name']}\n减重：{100 * best['mass_reduction']:.1f}%",
        (54, 58),
    )
    ax.set_xlabel("总质量")
    ax.set_ylabel("屈曲压力")
    ax.set_title("筒体优化验证结果")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=4.5, label="不可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=4.8, label="可行"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["black"], markersize=4.8, label="基准"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.5, label="最优解"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    beautify_axis(ax)

    save_figure(fig, "cylinder_mass_constraint_map")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_mass_constraint_map(baseline, design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_mass_constraint_map.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
