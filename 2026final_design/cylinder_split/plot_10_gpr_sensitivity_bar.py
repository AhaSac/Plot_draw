"""
plot_10_gpr_sensitivity_bar.py

输出 PNG：figures/cylinder/cylinder_gpr_sensitivity_bar.png
功能：GPR 置换敏感性柱状图：比较各设计参数对质量和屈曲压力的影响。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_gpr_sensitivity_bar() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_gpr_sensitivity_bar.png ==============================
# 功能：GPR 置换敏感性柱状图：比较各设计参数对质量和屈曲压力的影响。
def draw_gpr_sensitivity_bar(sensitivity: pd.DataFrame) -> None:
    """敏感性柱状图：展示置换效应与线性相关性。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), constrained_layout=True, sharey=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = sensitivity[sensitivity["target"] == target].sort_values("normalized_gp_effect", ascending=True)
        y = np.arange(len(subset))
        ax.barh(
            y,
            subset["normalized_gp_effect"],
            color=COLORS["blue"] if target == "total_mass" else COLORS["green"],
            edgecolor=COLORS["black"],
            linewidth=0.4,
            alpha=0.92,
            label="GPR 置换效应",
        )
        ax.scatter(
            subset["abs_pearson_correlation"],
            y,
            marker="D",
            s=18,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.3,
            label="|皮尔逊相关系数 r|",
            zorder=3,
        )
        ax.set_yticks(y, [PARAMETER_LABELS[p] for p in subset["parameter"]])
        ax.set_xlabel("归一化敏感性")
        ax.set_title(RESPONSE_LABELS[target])
        ax.set_xlim(0, max(1.0, subset[["normalized_gp_effect", "abs_pearson_correlation"]].max().max() * 1.08))
        beautify_axis(ax, minor=False)

    axes[1].legend(loc="lower right")
    fig.suptitle("设计参数敏感性分析", y=1.03, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_gpr_sensitivity_bar")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    sensitivity = gp_permutation_sensitivity(design, gpr_models)
    draw_gpr_sensitivity_bar(sensitivity)
    sensitivity_path = OUT_DIR / "cylinder_gpr_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_gpr_sensitivity_bar.png'}")
    print(f"Saved CSV: {sensitivity_path}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
