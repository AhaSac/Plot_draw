"""
plot_03_mass_constraint_map.py

输出 PNG：figures/head/封头质量-约束权衡.png
功能：质量-屈曲压力平面图：展示可行域、不可行点、基准和最优解。
"""

from head_plot_common import *
from matplotlib.lines import Line2D


# ============================== 对应 PNG：封头质量-约束权衡.png ==============================
# 功能：质量-屈曲压力平面图：展示可行域、不可行点、基准和最优解。
def draw_mass_constraint_map(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """质量-屈曲压力平面图。"""
    fig = plt.figure(figsize=(4.9, 2.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.34])

    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

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
    )
    ax.scatter(
        feasible["total_mass"],
        feasible["buckling_pressure"],
        s=27,
        color=COLORS["green"],
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
    )
    ax.scatter(
        float(baseline["total_mass"]),
        pressure_limit,
        marker="s",
        s=32,
        color=COLORS["blue"],
        edgecolors=COLORS["black"],
        linewidths=0.4,
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
        zorder=5,
    )

    annotate_best(
        ax,
        best["total_mass"],
        best["buckling_pressure"],
        f"减重：{100 * best['mass_reduction']:.1f}%",
        (54, 58),
    )

    ax.set_xlabel("总质量")
    ax.set_ylabel("屈曲特征值")
    ax.set_ylim(bottom=0)
    beautify_axis(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=5, label="不可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=5, label="可行"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["black"], markersize=5, label="基准解"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=8, label="最优解"),
        Line2D([0], [0], color=COLORS["black"], lw=0.85, ls=(0, (3, 2)), label="屈曲特征值阈值"),
    ]
    ax_legend.legend(
        handles=legend_handles,
        loc="center left",
        fontsize=7,
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        labelspacing=1.2,
        borderpad=1.0,
    )

    save_figure(fig, "封头质量-约束权衡")
    plt.close(fig)


def main() -> None:
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_mass_constraint_map(baseline, design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '封头质量-约束权衡.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
