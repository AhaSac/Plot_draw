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
    """平行坐标图：强化不可行点与 GPR 引导点的颜色区分。"""
    variables = PARAMETER_COLS + RESPONSE_COLS
    display_labels = [
        "t", "n", "p", "a",
        "质量",
        "屈曲特征值",
    ]
    normalized = design[variables].apply(normalize_to_unit)
    x = np.arange(len(variables))

    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)

    # 1. 绘制不可行点：使用极浅的灰色，线宽极细，作为背景
    for _, row in normalized.loc[~design["feasible"]].iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color="#AEAEAE", lw=0.4, alpha=0.15, zorder=1)

    # 2. 定义阶段颜色逻辑
    # 调整 GPR 的颜色为更亮、对比度更高的鲜绿色或深绿色
    stages_config = [
        ("LHS screening", COLORS["blue"], 0.4, "LHS 可行", 0.95),
        ("GPR-guided verification", "#2ECC71", 0.6, "GPR 引导可行", 1.25), # 使用更鲜亮的翡翠绿，加粗
    ]

    for stage, color, alpha, label, lw in stages_config:
        subset = design[(design["feasible"]) & (design["stage"] == stage)]
        for idx in subset.index:
            ax.plot(x, normalized.loc[idx].to_numpy(dtype=float), color=color, lw=lw, alpha=alpha, zorder=2)

    # 3. 最优解轨迹
    best_values = []
    for variable in variables:
        low = design[variable].min()
        high = design[variable].max()
        val = 0.0 if abs(high - low) < 1e-12 else (best[variable] - low) / (high - low)
        best_values.append(val)

    ax.plot(
        x, best_values,
        color=COLORS["orange"],
        lw=2.6,
        marker="o",
        markersize=5.0,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=f"最优解：{best['case_name']}",
        zorder=5,
    )

    # 辅助线与刻度标注
    for xpos in x:
        ax.axvline(xpos, color="#D0D0D0", lw=0.7, ls="-", zorder=0)

    for xpos, variable in zip(x, variables):
        low, high = design[variable].min(), design[variable].max()
        ax.text(xpos, 1.05, f"{high:.3g}", ha="center", va="bottom", fontsize=6, color="#666666")
        ax.text(xpos, -0.05, f"{low:.3g}", ha="center", va="top", fontsize=6, color="#666666")

    # 样式设置
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("归一化取值")
    
    # 4. 图例调整：将图例放在右侧，框线淡化
    # 明确列出各阶段颜色，确保图中曲线与图例一致
    legend_handles = [
        Line2D([0], [0], color="#E0E0E0", lw=1.0, alpha=0.6, label="不可行"),
        Line2D([0], [0], color=COLORS["blue"], lw=0.95, alpha=0.8, label="LHS 可行"),
        Line2D([0], [0], color="#2ECC71", lw=1.25, alpha=0.9, label="GPR 引导可行"),
        Line2D([0], [0], color=COLORS["orange"], marker="o", markersize=6,
               markeredgecolor="black", lw=2.5, label="最优解"),
    ]
    
    ax.legend(
        handles=legend_handles, 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=True,
        edgecolor="#EEEEEE"
    )

    beautify_axis(ax, minor=False)
    save_figure(fig, "参数平行优化变化")
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
