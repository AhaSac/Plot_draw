"""
plot_01_optimization_summary.py

输出 PNG：figures/cylinder/cylinder_optimization_summary.png
功能：综合总览图：迭代过程、质量-约束权衡、可行域与阶段最优。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_summary_figure() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_optimization_summary.png ==============================
# 功能：综合总览图：迭代过程、质量-约束权衡、可行域与阶段最优。
def draw_summary_figure(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """综合总览图：迭代过程、质量-约束权衡、可行域与阶段最优。"""
    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.95])

    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    ordered = running_best(design)
    infeasible = ordered[~ordered["feasible"]]
    feasible = ordered[ordered["feasible"]]

    ax0.scatter(
        infeasible["case_index"],
        infeasible["total_mass"],
        s=17,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
        label="不可行",
    )
    ax0.scatter(
        feasible["case_index"],
        feasible["total_mass"],
        s=22,
        color=COLORS["green"],
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        label="可行",
        zorder=3,
    )
    ax0.plot(
        ordered["case_index"],
        ordered["running_best_mass"],
        color=COLORS["orange"],
        lw=1.5,
        drawstyle="steps-post",
        label="当前最优可行解",
        zorder=2,
    )
    ax0.axhline(float(baseline["total_mass"]), color=COLORS["black"], lw=0.8, ls=(0, (3, 2)))
    ax0.scatter(
        best["case_index"],
        best["total_mass"],
        marker="*",
        s=120,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )
    annotate_best(
        ax0,
        best["case_index"],
        best["total_mass"],
        f"最优：{best['case_name']}\n质量 = {best['total_mass']:.4f}",
        (8, 12),
    )
    ax0.set_xlabel("已验证算例编号")
    ax0.set_ylabel("总质量")
    ax0.set_title("优化迭代过程")
    ax0.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax0)
    panel_label(ax0, "a")

    ax1.axhspan(pressure_limit, design["buckling_pressure"].max() * 1.08, color=COLORS["pale_green"], zorder=0)
    ax1.axhspan(0, pressure_limit, color=COLORS["pale_red"], zorder=0)
    ax1.axhline(
        pressure_limit,
        color=COLORS["black"],
        lw=0.85,
        ls=(0, (3, 2)),
        label=f"基准屈曲压力 = {pressure_limit:.5f}",
    )
    ax1.scatter(
        design.loc[~design["feasible"], "total_mass"],
        design.loc[~design["feasible"], "buckling_pressure"],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
    )
    sc1 = ax1.scatter(
        design.loc[design["feasible"], "total_mass"],
        design.loc[design["feasible"], "buckling_pressure"],
        c=design.loc[design["feasible"], "case_index"],
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    ax1.scatter(
        best["total_mass"],
        best["buckling_pressure"],
        marker="*",
        s=125,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )
    annotate_best(
        ax1,
        best["total_mass"],
        best["buckling_pressure"],
        f"减重 {100 * best['mass_reduction']:.1f}%\n屈曲压力 = {best['buckling_pressure']:.5f}",
        (18, 36),
    )
    cbar = fig.colorbar(sc1, ax=ax1, fraction=0.045, pad=0.02)
    cbar.set_label("算例编号")
    ax1.set_xlabel("总质量")
    ax1.set_ylabel("屈曲压力")
    ax1.set_title("质量-约束权衡")
    ax1.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax1)
    panel_label(ax1, "b")

    p_values = np.linspace(max(1.0, design["p"].min() * 0.9), design["p"].max() * 1.04, 300)
    n_limit = CYLINDER_LENGTH / p_values
    ax2.fill_between(
        p_values,
        n_limit,
        design["n"].max() + 0.8,
        color=COLORS["pale_red"],
        alpha=0.65,
        linewidth=0,
        label="n × p > 2650",
    )
    ax2.plot(p_values, n_limit, color=COLORS["black"], lw=0.9, ls=(0, (3, 2)), label="n × p = 2650")
    norm = Normalize(vmin=design["total_mass"].min(), vmax=design["total_mass"].max())
    sc2 = ax2.scatter(
        design["p"],
        design["n"],
        c=design["total_mass"],
        cmap="magma_r",
        norm=norm,
        s=np.where(design["feasible"], 31, 18),
        marker="o",
        edgecolors=np.where(design["feasible"], COLORS["black"], COLORS["gray"]),
        linewidths=np.where(design["feasible"], 0.35, 0.25),
        alpha=np.where(design["feasible"], 0.98, 0.55),
        zorder=3,
    )
    ax2.scatter(
        best["p"],
        best["n"],
        marker="*",
        s=125,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.045, pad=0.02)
    cbar2.set_label("总质量")
    ax2.set_xlabel("波距 p")
    ax2.set_ylabel("波纹数 n")
    ax2.set_title("设计空间可行域")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax2)
    panel_label(ax2, "c")

    stages = ["Baseline", "LHS screening", "GPR-guided verification"]
    values = [float(baseline["total_mass"])]
    labels = ["unoptimized"]
    for stage in stages[1:]:
        subset = design[(design["stage"] == stage) & (design["feasible"])]
        if subset.empty:
            values.append(np.nan)
            labels.append("none")
        else:
            row = subset.sort_values("total_mass").iloc[0]
            values.append(float(row["total_mass"]))
            labels.append(str(row["case_name"]))

    x = np.arange(len(stages))
    bar_colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"]]
    ax3.bar(x, values, color=bar_colors, width=0.58, edgecolor=COLORS["black"], linewidth=0.45)
    ax3.set_xticks(x, ["基准", "初始\nLHS", "GPR 引导\n验证"])
    ax3.set_ylabel("最优可行质量")
    ax3.set_title("各阶段最优已验证方案")
    for idx, value in enumerate(values):
        if np.isfinite(value):
            reduction = 1.0 - value / float(baseline["total_mass"])
            ax3.text(
                idx,
                value + 0.006,
                f"{value:.3f}\n({reduction:.0%})",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=COLORS["black"],
            )
    ax3.text(
        0.03,
        0.96,
        (
            f"当前最优解\n"
            f"{best['case_name']}\n"
            f"t={best['t']:.3f}, n={int(best['n'])}, p={int(best['p'])}, a={int(best['a'])}"
        ),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["black"],
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": COLORS["light_gray"], "lw": 0.6},
    )
    ax3.set_ylim(0, max(values) * 1.18)
    beautify_axis(ax3, minor=False)
    panel_label(ax3, "d")

    save_figure(fig, "cylinder_optimization_summary")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_summary_figure(baseline, design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_optimization_summary.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
