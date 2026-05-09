"""
plot_07_sampling_scatter_matrix.py

输出 PNG：figures/cylinder/cylinder_sampling_scatter_matrix.png
功能：采样散点矩阵：对比初始 LHS、GPR 引导点和随机参考分布。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_sampling_scatter_matrix() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_sampling_scatter_matrix.png ==============================
# 功能：采样散点矩阵：对比初始 LHS、GPR 引导点和随机参考分布。
def draw_sampling_scatter_matrix(design: pd.DataFrame, best: pd.Series) -> None:
    """采样散点矩阵：对比初始 LHS、GPR 引导点与随机参考分布。"""
    rng = np.random.default_rng(GPR_RANDOM_SEED + 3)
    lhs = design[design["stage"] == "LHS screening"]
    gpr = design[design["stage"] == "GPR-guided verification"]
    random_ref = random_design_samples(design, len(lhs), rng)

    n_vars = len(PARAMETER_COLS)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(6.6, 6.3), constrained_layout=True)

    for row_idx, y_col in enumerate(PARAMETER_COLS):
        for col_idx, x_col in enumerate(PARAMETER_COLS):
            ax = axes[row_idx, col_idx]

            if row_idx == col_idx:
                bins = 8 if x_col != "n" else np.arange(design[x_col].min() - 0.5, design[x_col].max() + 1.5, 1)
                ax.hist(
                    random_ref[x_col],
                    bins=bins,
                    color=COLORS["light_gray"],
                    edgecolor="white",
                    alpha=0.85,
                    density=True,
                    label="随机参考",
                )
                ax.hist(
                    lhs[x_col],
                    bins=bins,
                    histtype="step",
                    color=COLORS["blue"],
                    lw=1.1,
                    density=True,
                    label="初始 LHS",
                )
                ax.axvline(best[x_col], color=COLORS["orange"], lw=1.3)
            else:
                ax.scatter(
                    random_ref[x_col],
                    random_ref[y_col],
                    s=8,
                    color=COLORS["light_gray"],
                    alpha=0.55,
                    linewidths=0,
                )
                ax.scatter(
                    lhs[x_col],
                    lhs[y_col],
                    s=14,
                    color=COLORS["blue"],
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.85,
                )
                ax.scatter(
                    gpr[x_col],
                    gpr[y_col],
                    s=16,
                    facecolors="none",
                    edgecolors=COLORS["orange"],
                    linewidths=0.65,
                    alpha=0.88,
                )
                ax.scatter(
                    best[x_col],
                    best[y_col],
                    marker="*",
                    s=75,
                    color=COLORS["orange"],
                    edgecolors=COLORS["black"],
                    linewidths=0.35,
                    zorder=4,
                )

            if row_idx == n_vars - 1:
                ax.set_xlabel(PARAMETER_LABELS[x_col])
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(PARAMETER_LABELS[y_col])
            else:
                ax.set_yticklabels([])

            if x_col in INTEGER_PARAMETER_COLS and row_idx != col_idx:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if y_col in INTEGER_PARAMETER_COLS and row_idx != col_idx:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

            beautify_axis(ax, minor=False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["light_gray"], markeredgecolor=COLORS["light_gray"], markersize=4.5, label="随机参考"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", markersize=4.8, label="初始 LHS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["orange"], markersize=5.0, label="GPR 引导验证"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=7.5, label="最优解"),
    ]
    fig.suptitle("采样点在二维投影中的分布", y=1.02, fontsize=8.6, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=4, columnspacing=1.15)
    save_figure(fig, "cylinder_sampling_scatter_matrix")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_sampling_scatter_matrix(design, best)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_sampling_scatter_matrix.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
