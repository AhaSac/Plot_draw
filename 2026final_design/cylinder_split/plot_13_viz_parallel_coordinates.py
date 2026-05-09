"""
plot_13_viz_parallel_coordinates.py

输出 PNG：figures/cylinder/cylinder_viz_parallel_coordinates.png
功能：原 viz 三联图第 2 张：低质量/高质量方案的平行坐标对比。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_viz_parallel_coordinates() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_viz_parallel_coordinates.png ==============================
# 功能：原 viz 三联图第 2 张：低质量/高质量方案的平行坐标对比。
def draw_viz_parallel_coordinates() -> None:
    """PNG: cylinder_viz_parallel_coordinates.png；功能：viz 数据的平行坐标图，突出低质量/高质量方案。"""
    df = load_viz_dataframe()

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    dims = ["t", "n", "p", "a", "mass"]
    labels = ["t", "n", "p", "a", "mass"]
    norm_df = df[dims].copy()
    for dim in dims:
        norm_df[dim] = (df[dim] - df[dim].min()) / (df[dim].max() - df[dim].min())

    q1 = df["mass"].quantile(0.25)
    q3 = df["mass"].quantile(0.75)
    colors = []
    for mass in df["mass"]:
        if mass <= q1:
            colors.append("#1D9E75")
        elif mass >= q3:
            colors.append("#D85A30")
        else:
            colors.append("#999999")

    x_pos = np.arange(len(dims))
    for idx in range(len(df)):
        ax.plot(
            x_pos,
            norm_df.iloc[idx].values,
            color=colors[idx],
            alpha=0.6,
            linewidth=1.2 if colors[idx] == "#1D9E75" else 0.7,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([])
    ax.set_title("Parallel coordinates — green = low mass (best), red = high mass", fontsize=11, pad=10)
    for xpos in x_pos:
        ax.axvline(x=xpos, color="gray", alpha=0.25, linewidth=0.5)
    for xpos, dim in enumerate(dims):
        ax.text(xpos, 1.05, f"{df[dim].max():.2f}", ha="center", fontsize=7, color="#666")
        ax.text(xpos, -0.05, f"{df[dim].min():.2f}", ha="center", va="top", fontsize=7, color="#666")

    fig.savefig(OUT_DIR / "cylinder_viz_parallel_coordinates.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    draw_viz_parallel_coordinates()
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_viz_parallel_coordinates.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
