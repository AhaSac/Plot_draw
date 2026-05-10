"""
plot_04_parameter_response_grid.py

输出 PNG：figures/head/参数响应关系.png
功能：参数-响应关系网格图：展示 9 个设计变量与质量、屈曲压力的二维关联。
"""

from head_plot_common import *


# ============================== 对应 PNG：参数响应关系.png ==============================
# 功能：参数-响应关系网格图：展示 9 个设计变量与质量、屈曲压力的二维关联。
def draw_parameter_response_grid(
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """核心关系图：每个参数同时展示归一化质量和归一化屈曲压力趋势。"""
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.0), constrained_layout=True)

    infeasible = design[~design["feasible"]]
    feasible = design[design["feasible"]]
    mass_norm = normalize_to_unit(design["total_mass"])
    pressure_norm = normalize_to_unit(design["buckling_pressure"])

    legend_handles = None
    for idx, param in enumerate(PARAMETER_COLS):
        ax = axes.flat[idx]
        x_infeasible = infeasible[param]
        x_feasible = feasible[param]
        mass_infeasible = mass_norm.loc[infeasible.index]
        mass_feasible = mass_norm.loc[feasible.index]
        pressure_infeasible = pressure_norm.loc[infeasible.index]
        pressure_feasible = pressure_norm.loc[feasible.index]

        ax.scatter(x_infeasible, mass_infeasible, s=11, facecolors="white", edgecolors=COLORS["gray"], linewidths=0.4, alpha=0.55)
        ax.scatter(x_feasible, mass_feasible, s=13, color=COLORS["green"], edgecolors="white", linewidths=0.25, alpha=0.75)
        ax.scatter(x_infeasible, pressure_infeasible, s=11, facecolors="none", edgecolors=COLORS["blue"], linewidths=0.4, alpha=0.35)
        ax.scatter(x_feasible, pressure_feasible, s=13, color=COLORS["blue"], edgecolors="white", linewidths=0.25, alpha=0.65)



        ax.scatter(
            best[param],
            float(normalize_to_unit(design["total_mass"]).loc[best.name]),
            marker="*",
            s=75,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.35,
            zorder=5,
        )

        ax.set_title(PARAMETER_LABELS[param])
        ax.set_ylim(-0.05, 1.05)
        if param in INTEGER_PARAMETER_COLS:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        if idx >= 6:
            ax.set_xlabel(PARAMETER_LABELS[param])
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if idx % 3 == 0:
            ax.set_ylabel("归一化响应")
        else:
            ax.set_ylabel("")

        beautify_axis(ax)

        if legend_handles is None:
            legend_handles = [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=4.5, label="可行-质量"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=4.5, label="不可行-质量"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", markersize=4.5, label="可行-压力"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["blue"], markersize=4.5, label="不可行-压力"),
                Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.0, label="最优解"),
            ]

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.03), ncol=5, handlelength=1.2, columnspacing=1.0)
    fig.suptitle("参数-响应关系网格图", fontsize=8.0, y=1.07)
    save_figure(fig, "参数响应关系")
    plt.close(fig)


def main() -> None:
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_parameter_response_grid(design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '参数响应关系.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
