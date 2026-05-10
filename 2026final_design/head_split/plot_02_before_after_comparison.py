"""
plot_02_before_after_comparison.py

输出 PNG：figures/head/封头优化前后对比.png
功能：优化前后对比柱状图：展示质量下降和屈曲压力提升。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 head_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_before_after_comparison() 函数。
"""

from head_plot_common import *


# ============================== 对应 PNG：封头优化前后对比.png ==============================
# 功能：优化前后对比柱状图：展示质量下降和屈曲压力提升。
def draw_before_after_comparison(baseline: pd.Series, best: pd.Series) -> None:
    """单独输出“优化前 vs 优化后”对比柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), constrained_layout=True)

    reduction = 1.0 - float(best["total_mass"]) / float(baseline["total_mass"])
    pressure_gain = float(best["buckling_pressure"]) / float(baseline["buckling_pressure"]) - 1.0

    axes[0].bar(
        ["基准", "优化后"],
        [float(baseline["total_mass"]), float(best["total_mass"])],
        color=[COLORS["gray"], COLORS["orange"]],
        edgecolor=COLORS["black"],
        linewidth=0.5,
        width=0.58,
    )
    axes[0].set_title("总质量对比")
    axes[0].set_ylabel("总质量")
    axes[0].text(
        1,
        float(best["total_mass"]) + 0.002,
        f"减重 {100 * reduction:.1f}%",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    beautify_axis(axes[0], minor=False)

    axes[1].bar(
        ["基准", "优化后"],
        [float(baseline["buckling_pressure"]), float(best["buckling_pressure"])],
        color=[COLORS["blue"], COLORS["green"]],
        edgecolor=COLORS["black"],
        linewidth=0.5,
        width=0.58,
    )
    axes[1].set_title("屈曲压力对比")
    axes[1].set_ylabel("屈曲压力")
    axes[1].text(
        1,
        float(best["buckling_pressure"]) + 0.01 * float(best["buckling_pressure"]),
        f"提升 {100 * pressure_gain:.1f}%",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    beautify_axis(axes[1], minor=False)

    save_figure(fig, "封头优化前后对比")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_before_after_comparison(baseline, best)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '封头优化前后对比.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
