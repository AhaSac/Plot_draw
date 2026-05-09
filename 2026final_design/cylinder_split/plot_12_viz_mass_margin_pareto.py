"""
plot_12_viz_mass_margin_pareto.py

输出 PNG：figures/cylinder/cylinder_viz_mass_margin_pareto.png
功能：原 viz 三联图第 1 张：质量-屈曲裕度权衡散点图与 Pareto 前沿。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_viz_mass_margin_pareto() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_viz_mass_margin_pareto.png ==============================
# 功能：原 viz 三联图第 1 张：质量-屈曲裕度权衡散点图与 Pareto 前沿。
def draw_viz_mass_margin_pareto() -> None:
    """PNG: cylinder_viz_mass_margin_pareto.png；功能：质量-屈曲裕度散点图，并显示 Pareto 前沿。"""
    df = load_viz_dataframe()

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    scatter = ax.scatter(
        df["mass"],
        df["margin"],
        c=df["t"],
        cmap="viridis",
        s=df["n"] * 15,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    pareto_df = compute_pareto_front(df)
    ax.plot(
        pareto_df["mass"],
        pareto_df["margin"],
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
        label="Pareto front",
        zorder=1,
    )

    # 只标注关键方案，避免标签过密。
    for _, row in df.iterrows():
        if row["name"] in {"G51", "G13", "G12", "G11", "G24", "G14", "G03", "G01"}:
            ax.annotate(
                row["name"],
                (row["mass"], row["margin"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
                color="#444",
            )

    ax.axhline(y=0, color="red", linestyle=":", alpha=0.4, linewidth=1)
    ax.axhline(y=0.05, color="orange", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(0.36, 0.06, "safety threshold ~0.05", fontsize=8, color="orange", alpha=0.8)
    fig.colorbar(scatter, ax=ax, label="thickness t")
    ax.set_xlabel("Mass (minimize →)", fontsize=11)
    ax.set_ylabel("Buckling margin (must be ≥ 0)", fontsize=11)
    ax.set_title("Mass vs Buckling Margin — Pareto front shows best trade-offs", fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT_DIR / "cylinder_viz_mass_margin_pareto.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    draw_viz_mass_margin_pareto()
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_viz_mass_margin_pareto.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
