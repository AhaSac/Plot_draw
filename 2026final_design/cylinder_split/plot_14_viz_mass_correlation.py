"""
plot_14_viz_mass_correlation.py

输出 PNG：figures/cylinder/cylinder_viz_mass_correlation.png
功能：原 viz 三联图第 3 张：设计变量与质量的相关性条形图。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_viz_mass_correlation() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_viz_mass_correlation.png ==============================
# 功能：原 viz 三联图第 3 张：设计变量与质量的相关性条形图。
def draw_viz_mass_correlation() -> None:
    """PNG: cylinder_viz_mass_correlation.png；功能：展示 t/n/p/a 与质量 mass 的 Pearson 相关性。"""
    df = load_viz_dataframe()

    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    correlations = {var: df[var].corr(df["mass"]) for var in ["t", "n", "p", "a"]}
    vars_list = list(correlations.keys())
    corrs = [correlations[var] for var in vars_list]
    colors_bar = ["#D85A30" if corr > 0 else "#1D9E75" for corr in corrs]

    bars = ax.barh(vars_list, corrs, color=colors_bar, alpha=0.8, edgecolor="white")
    for bar, corr in zip(bars, corrs):
        ax.text(
            corr + (0.02 if corr > 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{corr:+.2f}",
            va="center",
            ha="left" if corr > 0 else "right",
            fontsize=10,
            fontweight="bold",
        )
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with mass", fontsize=11)
    ax.set_xlim(-1, 1)
    ax.set_title("Which variable drives mass the most?", fontsize=11, pad=10)
    ax.grid(True, alpha=0.3, axis="x")

    fig.savefig(OUT_DIR / "cylinder_viz_mass_correlation.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    draw_viz_mass_correlation()
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_viz_mass_correlation.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
