"""
plot_05_parallel_coordinates.py

输出 PNG：figures/cylinder/cylinder_parallel_coordinates.png
功能：平行坐标图：展示高维参数组合、响应指标和最优解轨迹。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_parallel_coordinates() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_parallel_coordinates.png ==============================
# 功能：平行坐标图：展示高维参数组合、响应指标和最优解轨迹。
def draw_parallel_coordinates(design: pd.DataFrame, best: pd.Series) -> None:
    """平行坐标图：展示高维参数组合及最优解轨迹。"""
    variables = PARAMETER_COLS + RESPONSE_COLS
    display_labels = [
        "t",
        "n",
        "p",
        "a",
        "质量\n越低越好",
        "屈曲压力\n越高越好",
    ]
    normalized = design[variables].apply(normalize_to_unit)
    x = np.arange(len(variables))

    fig, ax = plt.subplots(figsize=(6.85, 2.95), constrained_layout=True)

    for _, row in normalized.loc[~design["feasible"]].iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color=COLORS["gray"], lw=0.55, alpha=0.25, zorder=1)

    for stage, color, alpha, label in [
        ("LHS screening", COLORS["blue"], 0.34, "Feasible LHS"),
        ("GPR-guided verification", COLORS["green"], 0.45, "Feasible GPR-guided"),
    ]:
        subset = design[(design["feasible"]) & (design["stage"] == stage)]
        for idx in subset.index:
            ax.plot(x, normalized.loc[idx].to_numpy(dtype=float), color=color, lw=0.95, alpha=alpha, zorder=2)

    best_values = []
    for variable in variables:
        low = design[variable].min()
        high = design[variable].max()
        if abs(high - low) < 1e-12:
            best_values.append(0.0)
        else:
            best_values.append((best[variable] - low) / (high - low))

    ax.plot(
        x,
        best_values,
        color=COLORS["orange"],
        lw=2.4,
        marker="o",
        markersize=4.6,
        markeredgecolor=COLORS["black"],
        markeredgewidth=0.35,
        label=f"最优解：{best['case_name']}",
        zorder=5,
    )

    for xpos in x:
        ax.axvline(xpos, color=COLORS["light_gray"], lw=0.65, zorder=0)

    for xpos, variable in zip(x, variables):
        low = design[variable].min()
        high = design[variable].max()
        ax.text(xpos, 1.055, f"{high:.3g}", ha="center", va="bottom", fontsize=5.8, color=COLORS["gray"])
        ax.text(xpos, -0.075, f"{low:.3g}", ha="center", va="top", fontsize=5.8, color=COLORS["gray"])

    ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    ax.set_ylim(-0.11, 1.11)
    ax.set_xticks(x, display_labels)
    ax.set_ylabel("归一化取值")
    ax.set_title("设计变量与响应的平行坐标图")
    legend_handles = [
        Line2D([0], [0], color=COLORS["gray"], lw=0.9, alpha=0.45, label="不可行"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.2, alpha=0.75, label="LHS 可行点"),
        Line2D([0], [0], color=COLORS["green"], lw=1.2, alpha=0.75, label="GPR 引导可行点"),
        Line2D([0], [0], color=COLORS["orange"], marker="o", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], lw=2.4, label="最优解"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    beautify_axis(ax, minor=False)
    save_figure(fig, "cylinder_parallel_coordinates")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_parallel_coordinates(design, best)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_parallel_coordinates.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
