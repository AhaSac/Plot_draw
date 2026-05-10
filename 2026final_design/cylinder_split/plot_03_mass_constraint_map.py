from cylinder_plot_common import *
from matplotlib.lines import Line2D

# ============================== 质量单位设置 ==============================
MASS_SCALE = 1000.0
MASS_UNIT = "kg"

def mass_to_kg(value):
    """将质量从 t 换算为 kg。"""
    return value * MASS_SCALE

# ============================== 绘图函数 ==============================
def draw_mass_constraint_map(
    baseline: pd.Series,
    design: pd.DataFrame,
    best: pd.Series,
    pressure_limit: float,
) -> None:
    """质量-屈曲压力平面图：图例独立放在右侧。"""
    
    # 创建画布，增加宽度比例 (width_ratios) 给右侧图例留出空间
    # figsize 宽度从 3.55 增加到 4.8，以容纳右侧框
    fig = plt.figure(figsize=(4.8, 2.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.35]) 
    
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off') # 隐藏右侧子图的坐标轴，仅作占位

    feasible = design[design["feasible"]]
    infeasible = design[~design["feasible"]]

    # 1. 主图绘制内容
    ax.axhspan(pressure_limit, design["buckling_pressure"].max() * 1.08, color=COLORS["pale_green"], zorder=0)
    ax.axhspan(0, pressure_limit, color=COLORS["pale_red"], zorder=0)
    ax.axhline(pressure_limit, color=COLORS["black"], lw=0.85, ls=(0, (3, 2)))

    ax.scatter(mass_to_kg(infeasible["total_mass"]), infeasible["buckling_pressure"], 
               s=18, facecolors="white", edgecolors=COLORS["gray"], linewidths=0.65, alpha=0.82)
    ax.scatter(mass_to_kg(feasible["total_mass"]), feasible["buckling_pressure"], 
               s=27, color=COLORS["green"], edgecolors="white", linewidths=0.35, alpha=0.95)
    ax.scatter(mass_to_kg(float(baseline["total_mass"])), pressure_limit, 
               marker="s", s=32, color=COLORS["blue"], edgecolors=COLORS["black"], linewidths=0.4, zorder=4)
    ax.scatter(mass_to_kg(best["total_mass"]), best["buckling_pressure"], 
               marker="*", s=130, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.45, zorder=5)

    # 标注最优解
    annotate_best(ax, mass_to_kg(best["total_mass"]), best["buckling_pressure"],
                  f"减重：{100 * best['mass_reduction']:.1f}%", (54, 58))

    ax.set_xlabel(f"总质量 ({MASS_UNIT})")
    ax.set_ylabel("屈曲特征值")
    ax.set_ylim(bottom=0) 

    # 2. 自定义图例句柄
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=5, label="不可行"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=5, label="可行"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["black"], markersize=5, label="未优化解"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["black"], markersize=8, label="最优解"),
        Line2D([0], [0], color=COLORS["black"], lw=0.85, ls=(0, (3, 2)), label="屈曲特征值约束")
    ]

    # 3. 将图例画在右侧专门的 ax_legend 上
    # 使用 loc='center' 居中，或者 'upper left' 靠顶
    ax_legend.legend(
        handles=legend_handles, 
        loc="center left", 
        fontsize=7,
        frameon=True,      # 开启边框
        edgecolor='#CCCCCC',
        facecolor='white',
        labelspacing=1.2,  # 竖排间距拉大一点，方便截图
        borderpad=1.0      # 增加内边距
    )
    
    beautify_axis(ax)
    save_figure(fig, "圆筒质量约束")
    plt.close(fig)

def main() -> None:
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_mass_constraint_map(baseline, design, best, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_mass_constraint_map.png'}")

if __name__ == "__main__":
    main()