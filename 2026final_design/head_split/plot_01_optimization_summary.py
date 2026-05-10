"""
plot_01_optimization_summary.py

输出 PNG：
- figures/head/01_优化迭代过程.png
- figures/head/02_质量-约束权衡.png
- figures/head/03_设计空间可行域.png
- figures/head/04_各阶段最优已验证方案.png

功能：将原来的 1 张综合总览图拆成 4 张独立图片。

修改提示：
- 本脚本只负责生成上面这 4 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 head_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方对应的 draw_*() 函数。
"""

from head_plot_common import *


SUMMARY_OUTPUTS = {
    "iteration": "01_优化迭代过程",
    "tradeoff": "02_质量-约束权衡",
    "feasible_region": "03_设计空间可行域",
    "stage_best": "04_各阶段最优已验证方案",
}


def draw_iteration_process(baseline: pd.Series, design: pd.DataFrame, best: pd.Series) -> None:
    """输出优化迭代过程图。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)

    ordered = running_best(design)
    infeasible = ordered[~ordered["feasible"]]
    feasible = ordered[ordered["feasible"]]

    ax.scatter(
        infeasible["case_index"],
        infeasible["total_mass"],
        s=17,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
        label="不可行",
    )
    ax.scatter(
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
    ax.plot(
        ordered["case_index"],
        ordered["running_best_mass"],
        color=COLORS["orange"],
        lw=1.5,
        drawstyle="steps-post",
        label="当前最优可行解",
        zorder=2,
    )
    ax.axhline(float(baseline["total_mass"]), color=COLORS["black"], lw=0.8, ls=(0, (3, 2)))
    ax.scatter(
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
        ax,
        best["case_index"],
        best["total_mass"],
        f"质量 = {best['total_mass']:.4f}",
        (8, 12),
    )
    ax.set_xlabel("已验证算例编号")
    ax.set_ylabel("总质量")
    ax.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax)

    save_figure(fig, SUMMARY_OUTPUTS["iteration"])
    plt.close(fig)


def draw_mass_tradeoff(baseline: pd.Series, design: pd.DataFrame, best: pd.Series, pressure_limit: float) -> None:
    """输出质量-约束权衡图。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)

    y_bottom = 0
    y_top = max(
        design["buckling_pressure"].max(),
        best["buckling_pressure"],
        pressure_limit,
    ) * 1.18

    ax.axhspan(pressure_limit, y_top, color=COLORS["pale_green"], zorder=0)
    ax.axhspan(y_bottom, pressure_limit, color=COLORS["pale_red"], zorder=0)

    ax.axhline(
        pressure_limit,
        color=COLORS["black"],
        lw=0.90,
        ls=(0, (2, 2)),
        label=f"基准屈曲特征值 = {pressure_limit:.5f}",
    )

    ax.scatter(
        design.loc[~design["feasible"], "total_mass"],
        design.loc[~design["feasible"], "buckling_pressure"],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
    )

    sc = ax.scatter(
        design.loc[design["feasible"], "total_mass"],
        design.loc[design["feasible"], "buckling_pressure"],
        c=design.loc[design["feasible"], "case_index"],
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    ax.scatter(
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
        ax,
        best["total_mass"],
        best["buckling_pressure"],
        f"减重 {100 * best['mass_reduction']:.1f}%\n屈曲特征值 = {best['buckling_pressure']:.5f}",
        (18, 36),
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("算例编号")

    ax.set_xlabel("总质量")
    ax.set_ylabel("屈曲特征值")

    ax.set_ylim(y_bottom, y_top)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
        handlelength=1.6,
    )

    beautify_axis(ax)

    save_figure(fig, SUMMARY_OUTPUTS["tradeoff"])
    plt.close(fig)


def draw_feasible_region(design: pd.DataFrame, best: pd.Series) -> None:
    """输出设计空间可行域图。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)

    x_col, y_col = "longitude_num", "latitude_num"
    x_values = np.linspace(design[x_col].min() - 0.5, design[x_col].max() + 0.5, 120)
    y_values = np.linspace(design[y_col].min() - 0.5, design[y_col].max() + 0.5, 120)
    xx, yy = np.meshgrid(x_values, y_values)

    ax.scatter(
        design.loc[~design["feasible"], x_col],
        design.loc[~design["feasible"], y_col],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
        label="不可行",
    )
    sc = ax.scatter(
        design.loc[design["feasible"], x_col],
        design.loc[design["feasible"], y_col],
        c=design.loc[design["feasible"], "total_mass"],
        cmap="magma_r",
        s=28,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
        label="可行",
    )
    ax.scatter(
        best[x_col],
        best[y_col],
        marker="*",
        s=125,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
        label="最优解",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("总质量")
    ax.set_xlabel(PARAMETER_LABELS[x_col])
    ax.set_ylabel(PARAMETER_LABELS[y_col])
    ax.set_title("设计空间可行域")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", handlelength=1.6)
    beautify_axis(ax)
    ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.0, 0.92),
    handlelength=1.6,
    )
    save_figure(fig, SUMMARY_OUTPUTS["feasible_region"])
    plt.close(fig)


def draw_stage_best_summary(baseline: pd.Series, design: pd.DataFrame, best: pd.Series) -> None:
    """输出各阶段最优已验证方案图。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)

    stages = ["基准", "前期筛选", "后期验证"]
    values = [float(baseline["total_mass"])]
    for stage in stages[1:]:
        subset = design[(design["stage"] == stage) & (design["feasible"])]
        if subset.empty:
            values.append(np.nan)
        else:
            row = subset.sort_values("total_mass").iloc[0]
            values.append(float(row["total_mass"]))

    x = np.arange(len(stages))
    bar_colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"]]
    ax.bar(x, values, color=bar_colors, width=0.58, edgecolor=COLORS["black"], linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(["基准", "LHS", "GPR"])
    ax.set_ylabel("最优可行质量")
    for idx, value in enumerate(values):
        if np.isfinite(value):
            reduction = 1.0 - value / float(baseline["total_mass"])
            ax.text(
                idx,
                value + 0.002,
                f"{value:.3f}\n({reduction:.0%})",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=COLORS["black"],
            )

    ax.set_ylim(0, max(values) * 1.18)
    beautify_axis(ax, minor=False)

    save_figure(fig, SUMMARY_OUTPUTS["stage_best"])
    plt.close(fig)


def draw_summary_figure(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """将原来的综合总览图拆成 4 张独立 PNG。"""
    (OUT_DIR / "cylinder_optimization_summary.png").unlink(missing_ok=True)
    draw_iteration_process(baseline, design, best)
    draw_mass_tradeoff(baseline, design, best, pressure_limit)
    draw_feasible_region(design, best)
    draw_stage_best_summary(baseline, design, best)


def main() -> None:
    """单独运行本文件时，生成 4 张中文命名的 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_summary_figure(baseline, design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['iteration'] + '.png')}")
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['tradeoff'] + '.png')}")
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['feasible_region'] + '.png')}")
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['stage_best'] + '.png')}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
