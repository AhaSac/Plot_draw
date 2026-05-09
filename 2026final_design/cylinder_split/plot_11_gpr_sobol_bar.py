"""
plot_11_gpr_sobol_bar.py

输出 PNG：figures/cylinder/cylinder_gpr_sobol_bar.png
功能：Sobol 全局敏感性柱状图：并排展示一阶指数 S1 与总阶指数 ST。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_gpr_sobol_bar() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_gpr_sobol_bar.png ==============================
# 功能：Sobol 全局敏感性柱状图：并排展示一阶指数 S1 与总阶指数 ST。
def draw_gpr_sobol_bar(sobol_df: pd.DataFrame) -> None:
    """Sobol 敏感性柱状图：并排展示 S1 与 ST。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True, sharey=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = sobol_df[sobol_df["target"] == target].copy()
        subset = subset.set_index("parameter").reindex(PARAMETER_COLS).reset_index()
        y = np.arange(len(subset))
        h = 0.34

        ax.barh(
            y - h / 2,
            subset["sobol_s1"],
            height=h,
            color=COLORS["blue"],
            edgecolor=COLORS["black"],
            linewidth=0.4,
            alpha=0.9,
            label="Sobol 一阶指数 S1",
        )
        ax.barh(
            y + h / 2,
            subset["sobol_st"],
            height=h,
            color=COLORS["green"],
            edgecolor=COLORS["black"],
            linewidth=0.4,
            alpha=0.9,
            label="Sobol 总阶指数 ST",
        )

        ax.set_yticks(y, [PARAMETER_LABELS[p] for p in subset["parameter"]])
        ax.set_xlabel("Sobol 指数")
        ax.set_xlim(0, 1.0)
        ax.set_title(RESPONSE_LABELS[target])
        beautify_axis(ax, minor=False)

    axes[1].legend(loc="lower right")
    fig.suptitle("Sobol 全局敏感性分析（基于 GPR 代理模型）", y=1.03, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_gpr_sobol_bar")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    sobol_df = sobol_sensitivity_from_gp(design, gpr_models)
    draw_gpr_sobol_bar(sobol_df)
    sobol_path = OUT_DIR / "cylinder_gpr_sobol_indices.csv"
    sobol_df.to_csv(sobol_path, index=False)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_gpr_sobol_bar.png'}")
    print(f"Saved CSV: {sobol_path}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
