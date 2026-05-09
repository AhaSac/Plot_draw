"""
plot_04_parameter_response_grid.py

输出 PNG：figures/cylinder/cylinder_parameter_response_grid.png
功能：参数-响应关系网格图：展示 t/n/p/a 与质量、屈曲压力的二维关联。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_parameter_response_grid() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_parameter_response_grid.png ==============================
# 功能：参数-响应关系网格图：展示 t/n/p/a 与质量、屈曲压力的二维关联。
def draw_parameter_response_grid(
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """核心关系图：t/n/p/a 与 mass/buckling_pressure 的二维关联。"""
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 3.65), constrained_layout=True)

    infeasible = design[~design["feasible"]]
    feasible = design[design["feasible"]]

    for col_idx, param in enumerate(PARAMETER_COLS):
        for row_idx, response in enumerate(RESPONSE_COLS):
            ax = axes[row_idx, col_idx]

            ax.scatter(
                infeasible[param],
                infeasible[response],
                s=16,
                facecolors="white",
                edgecolors=COLORS["gray"],
                linewidths=0.55,
                alpha=0.72,
                zorder=2,
            )
            ax.scatter(
                feasible[param],
                feasible[response],
                s=24,
                color=COLORS["green"],
                edgecolors="white",
                linewidths=0.35,
                alpha=0.94,
                zorder=3,
            )

            gpr_round = design[design["stage"] == "GPR-guided verification"]
            ax.scatter(
                gpr_round[param],
                gpr_round[response],
                s=32,
                facecolors="none",
                edgecolors=COLORS["orange"],
                linewidths=0.55,
                alpha=0.72,
                zorder=4,
            )

            x_all, y_all = binned_median_line(design[param], design[response])
            if len(x_all):
                ax.plot(x_all, y_all, color=COLORS["gray"], lw=0.9, ls=(0, (2, 2)), zorder=1)

            x_feas, y_feas = binned_median_line(feasible[param], feasible[response])
            if len(x_feas):
                ax.plot(x_feas, y_feas, color=COLORS["green"], lw=1.2, zorder=5)

            ax.scatter(
                best[param],
                best[response],
                marker="*",
                s=95,
                color=COLORS["orange"],
                edgecolors=COLORS["black"],
                linewidths=0.4,
                zorder=6,
            )

            if response == "buckling_pressure":
                ax.axhline(pressure_limit, color=COLORS["black"], lw=0.75, ls=(0, (3, 2)), zorder=0)

            if row_idx == 1:
                ax.set_xlabel(PARAMETER_LABELS[param])
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(RESPONSE_LABELS[response])
            else:
                ax.set_ylabel("")

            if param in {"n", "p", "a"}:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            beautify_axis(ax)

    axes[0, 0].set_title("参数-响应关系图", loc="left", fontsize=8.2, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=4.2, label="不可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=4.8, label="可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["orange"], markersize=5.0, label="GPR 验证轮"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.5, label="最优解"),
        Line2D([0], [0], color=COLORS["green"], lw=1.2, label="可行点中位趋势"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.04),
        ncol=5,
        handlelength=1.5,
        columnspacing=1.2,
    )
    save_figure(fig, "cylinder_parameter_response_grid")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_parameter_response_grid(design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_parameter_response_grid.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
