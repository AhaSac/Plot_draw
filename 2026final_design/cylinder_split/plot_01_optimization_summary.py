"""
plot_01_optimization_summary.py

输出 PNG：
- figures/cylinder/01_优化迭代过程.png
- figures/cylinder/02_质量-约束权衡.png
- figures/cylinder/03_设计空间可行域.png
- figures/cylinder/04_各阶段最优已验证方案.png

功能：将原来的 1 张综合总览图拆成 4 张独立图片。

修改提示：
- 本脚本只负责生成上面这 4 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方对应的 draw_*() 函数。
"""

from cylinder_plot_common import *
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


SUMMARY_OUTPUTS = {
    "iteration": "01_优化迭代过程",
    "tradeoff": "02_质量-约束权衡",
    "feasible_region": "03_设计空间可行域",
    "stage_best": "04_各阶段最优已验证方案",
}


# 质量单位换算：原始数据按 t 处理，绘图时统一换成 kg
MASS_SCALE = 1000.0
MASS_UNIT = "kg"


def mass_to_kg(value):
    """将质量从 t 换算为 kg。支持标量、Series、ndarray。"""
    return value * MASS_SCALE


def make_axes_with_right_panel(cbar_width="22%", cbar_height="62%"):
    """
    创建主图轴 + 右侧辅助区域。
    右侧辅助区域上方放图例，下方放缩短后的色条。
    """
    fig = plt.figure(figsize=(5.15, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.34],
        wspace=0.03,
    )

    ax = fig.add_subplot(gs[0, 0])
    side_ax = fig.add_subplot(gs[0, 1])
    side_ax.set_axis_off()

    # 右侧色条：放在右侧区域下方，通过参数调整大小以减少上方留白
    cax = inset_axes(
        side_ax,
        width=cbar_width,
        height=cbar_height,
        loc="lower center",
        borderpad=0.0,
    )

    return fig, ax, side_ax, cax


def draw_iteration_process(baseline: pd.Series, design: pd.DataFrame, best: pd.Series) -> None:
    """输出优化迭代过程图。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)

    ordered = running_best(design)
    infeasible = ordered[~ordered["feasible"]]
    feasible = ordered[ordered["feasible"]]

    ax.scatter(
        infeasible["case_index"],
        mass_to_kg(infeasible["total_mass"]),
        s=17,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
        label="不可行",
    )

    ax.scatter(
        feasible["case_index"],
        mass_to_kg(feasible["total_mass"]),
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
        mass_to_kg(ordered["running_best_mass"]),
        color=COLORS["orange"],
        lw=1.5,
        drawstyle="steps-post",
        label="当前最优可行解",
        zorder=2,
    )

    # 黑色虚线：未优化筒体质量
    ax.axhline(
        mass_to_kg(float(baseline["total_mass"])),
        color=COLORS["black"],
        lw=0.8,
        ls=(0, (3, 2)),
    )

    ax.scatter(
        best["case_index"],
        mass_to_kg(best["total_mass"]),
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
        mass_to_kg(best["total_mass"]),
        f"质量 = {mass_to_kg(best['total_mass']):.2f} {MASS_UNIT}",
        (8, 12),
    )

    ax.set_xlabel("优化算例数量")
    ax.set_ylabel(f"总质量（{MASS_UNIT}）")

    # 在图例中补充黑色虚线说明，但不改变图中的黑线
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color=COLORS["black"],
            lw=0.8,
            ls=(0, (3, 2)),
        )
    )
    labels.append("未优化筒体质量")

    ax.legend(handles, labels, loc="upper right", handlelength=1.6)
    beautify_axis(ax)

    save_figure(fig, SUMMARY_OUTPUTS["iteration"])
    plt.close(fig)
def draw_mass_tradeoff(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """输出质量-约束权衡图。"""

    # 固定图中基准屈曲特征值虚线位置
    baseline_pressure_y = 0.5

    fig, ax, side_ax, cax = make_axes_with_right_panel(
        cbar_width="35%",
        cbar_height="82%",
    )

    # 背景上限，保证绿色阴影区域一定能覆盖到 0.5 以上
    y_top = max(
        design["buckling_pressure"].max() * 1.08,
        baseline_pressure_y * 1.12,
    )

    # 0.5 以上：满足约束区域
    ax.axhspan(
        baseline_pressure_y,
        y_top,
        color=COLORS["pale_green"],
        zorder=0,
    )

    # 0 到 0.5：不满足约束区域
    ax.axhspan(
        0,
        baseline_pressure_y,
        color=COLORS["pale_red"],
        zorder=0,
    )

    # 黑色虚线移动到 y = 0.5
    ax.axhline(
        baseline_pressure_y,
        color=COLORS["black"],
        lw=0.85,
        ls=(0, (3, 2)),
        label=f"基准屈曲特征值\n= {baseline_pressure_y:.5f}",
    )

    ax.scatter(
        mass_to_kg(design.loc[~design["feasible"], "total_mass"]),
        design.loc[~design["feasible"], "buckling_pressure"],
        s=18,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.65,
        alpha=0.78,
    )

    feasible_design = design[design["feasible"]].copy()

    case_norm = PowerNorm(
        gamma=0.45,
        vmin=feasible_design["case_index"].min(),
        vmax=feasible_design["case_index"].max(),
    )

    sc = ax.scatter(
        mass_to_kg(feasible_design["total_mass"]),
        feasible_design["buckling_pressure"],
        c=feasible_design["case_index"],
        cmap="turbo",
        norm=case_norm,
        s=50,
        edgecolors="white",
        linewidths=0.35,
        zorder=1,
    )

    ax.scatter(
        mass_to_kg(best["total_mass"]),
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
        mass_to_kg(best["total_mass"]),
        best["buckling_pressure"],
        f"减重 {100 * best['mass_reduction']:.1f}%\n屈曲特征值 = {best['buckling_pressure']:.5f}",
        (18, 36),
    )

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("算例编号")

    ax.set_xlabel(f"总质量（{MASS_UNIT}）")
    ax.set_ylabel("屈曲特征值")

    # Y 轴从 0 开始，同时保证能显示完整阴影范围
    ax.set_ylim(bottom=0, top=y_top)

    handles, labels = ax.get_legend_handles_labels()
    side_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.05, 1.03),
        handlelength=1.5,
        borderpad=0.2,
        labelspacing=0.2,
        fontsize=6.4,
        frameon=False,
    )

    beautify_axis(ax)

    save_figure(fig, SUMMARY_OUTPUTS["tradeoff"])
    plt.close(fig)
    
def draw_feasible_region(
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """输出设计空间可行域图。"""
    # 【修改1】调用新参数放大色表（与图02保持一致）
    fig, ax, side_ax, cax = make_axes_with_right_panel(cbar_width="35%", cbar_height="82%")

    p_values = np.linspace(
        max(1.0, design["p"].min() * 0.9),
        design["p"].max() * 1.04,
        300,
    )
    n_limit = CYLINDER_LENGTH / p_values

    y_bottom = design["n"].min() - 0.3
    y_top = design["n"].max() + 0.8

    # 阴影区：n × p > 2650
    # 将阴影下边界裁剪到图框上边界，避免阴影跑到黑色虚线下方，
    # 同时避免虚线靠近图框顶部时出现缺块。
    shade_lower = np.minimum(n_limit, y_top)

    ax.fill_between(
        p_values,
        shade_lower,
        y_top,
        color=COLORS["pale_red"],
        alpha=0.65,
        linewidth=0,
        label="n × p > 2650",
    )

    ax.plot(
        p_values,
        n_limit,
        color=COLORS["black"],
        lw=0.9,
        ls=(0, (3, 2)),
        label="n × p = 2650",
    )

    # 只显示同时满足以下条件的点：
    # 1. 屈曲压力满足要求
    # 2. 几何约束 n × p <= 2650
    plot_design = design[
        (design["buckling_pressure"] >= pressure_limit)
        & (design["n"] * design["p"] <= CYLINDER_LENGTH)
    ].copy()

    if plot_design.empty:
        print("Warning: 图03没有满足屈曲压力和几何约束的设计点。")
        cax.set_axis_off()
    else:
        plot_mass_kg = mass_to_kg(plot_design["total_mass"])

        norm = Normalize(
            vmin=plot_mass_kg.min(),
            vmax=plot_mass_kg.max(),
        )

        sc = ax.scatter(
            plot_design["p"],
            plot_design["n"],
            c=plot_mass_kg,
            cmap="magma_r",
            norm=norm,
            s=31,
            marker="o",
            edgecolors=COLORS["black"],
            linewidths=0.35,
            alpha=0.98,
            zorder=3,
        )

        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(f"总质量（{MASS_UNIT}）")

    ax.scatter(
        best["p"],
        best["n"],
        marker="*",
        s=125,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=0.45,
        zorder=5,
    )

    ax.set_xlabel("波距 p")
    ax.set_ylabel("波纹数 n")
    ax.set_ylim(y_bottom, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 【修改2】图例稍微向左上方移动，给变大的色条让出空间
    handles, labels = ax.get_legend_handles_labels()
    side_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.05, 1.03), 
        handlelength=1.5,
        borderpad=0.2,
        labelspacing=0.45,
        fontsize=6.4,
        frameon=False,
    )

    beautify_axis(ax)

    save_figure(fig, SUMMARY_OUTPUTS["feasible_region"])
    plt.close(fig)


def draw_stage_best_summary(baseline: pd.Series, design: pd.DataFrame, best: pd.Series) -> None:
    """输出各阶段最优已验证方案图。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)

    stages = ["Baseline", "LHS screening", "GPR-guided verification"]
    baseline_mass_kg = mass_to_kg(float(baseline["total_mass"]))
    values = [baseline_mass_kg]

    for stage in stages[1:]:
        subset = design[(design["stage"] == stage) & (design["feasible"])]
        if subset.empty:
            values.append(np.nan)
        else:
            row = subset.sort_values("total_mass").iloc[0]
            values.append(mass_to_kg(float(row["total_mass"])))

    x = np.arange(len(stages))
    bar_colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"]]

    ax.bar(
        x,
        values,
        color=bar_colors,
        width=0.58,
        edgecolor=COLORS["black"],
        linewidth=0.45,
    )

    ax.set_xticks(x, ["未优化算例", "LHS", "GPR"])
    ax.set_ylabel(f"最优可行解质量（{MASS_UNIT}）")

    for idx, value in enumerate(values):
        if np.isfinite(value):
            reduction = 1.0 - value / baseline_mass_kg
            ax.text(
                idx,
                value + 6.0,
                f"{value:.1f}\n减重({reduction:.0%})",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=COLORS["black"],
            )

    ax.set_ylim(0, np.nanmax(values) * 1.18)
    beautify_axis(ax, minor=False)

    save_figure(fig, SUMMARY_OUTPUTS["stage_best"])
    plt.close(fig)


# ============================== 对应 4 张 PNG（中文命名） ==============================
# 功能：将原来的综合总览图拆成 4 张独立 PNG。
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
    draw_feasible_region(design, best, pressure_limit)
    draw_stage_best_summary(baseline, design, best)


def main() -> None:
    """单独运行本文件时，生成 4 张中文命名的 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()

    _, baseline, design, best, _ = load_data()
    pressure_limit = 0.5

    draw_summary_figure(baseline, design, best, pressure_limit)

    cleanup_temp_files()

    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['iteration'] + '.png')}")
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['tradeoff'] + '.png')}")
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['feasible_region'] + '.png')}")
    print(f"Saved: {OUT_DIR / (SUMMARY_OUTPUTS['stage_best'] + '.png')}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()