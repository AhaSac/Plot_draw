"""
plot_06_parameter_evolution_heatmap.py

输出 PNG：figures/cylinder/cylinder_parameter_evolution_heatmap.png
功能：演化热图：观察参数与响应随优化迭代过程的变化。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_parameter_evolution_heatmap() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_parameter_evolution_heatmap.png ==============================
# 功能：演化热图：观察参数与响应随优化迭代过程的变化。
def draw_parameter_evolution_heatmap(design: pd.DataFrame, best: pd.Series) -> None:
    """演化热图：观察参数与响应随迭代过程的变化。"""
    ordered = design.sort_values("case_index").reset_index(drop=True)
    variables = ["t", "n", "p", "a", "n_times_p", "total_mass", "buckling_pressure"]
    labels = ["t", "n", "p", "a", "n × p", "质量", "屈曲压力"]
    matrix = ordered[variables].apply(normalize_to_unit).T.to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.2, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.0])
    ax_strip = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_strip)

    stage_strip = np.where(ordered["stage"] == "LHS screening", 0.35, 0.80)[None, :]
    ax_strip.imshow(stage_strip, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    feasible_x = np.where(ordered["feasible"].to_numpy())[0]
    infeasible_x = np.where(~ordered["feasible"].to_numpy())[0]
    ax_strip.scatter(infeasible_x, np.zeros_like(infeasible_x), s=8, color=COLORS["gray"], alpha=0.45)
    ax_strip.scatter(feasible_x, np.zeros_like(feasible_x), s=10, color=COLORS["green"], alpha=0.85)
    best_pos = int(np.where(ordered["case_name"].to_numpy() == best["case_name"])[0][0])
    ax_strip.scatter([best_pos], [0], marker="*", s=75, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.35, zorder=4)
    ax_strip.axvline(INITIAL_LHS_LAST_CASE - 0.5, color=COLORS["black"], lw=0.8, ls=(0, (3, 2)))
    ax_strip.text(
        0.24,
        0.5,
        "LHS 初始采样",
        transform=ax_strip.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["black"],
    )
    ax_strip.text(
        0.74,
        0.5,
        "GPR 引导验证",
        transform=ax_strip.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color="white",
    )
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("阶段", rotation=0, ha="right", va="center", fontsize=6.4)
    ax_strip.tick_params(axis="x", labelbottom=False, bottom=False, top=False)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(INITIAL_LHS_LAST_CASE - 0.5, color="white", lw=1.0, ls=(0, (3, 2)))
    ax.scatter([best_pos], [variables.index("total_mass")], marker="*", s=90, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.35, zorder=4)
    ax.scatter([best_pos], [variables.index("buckling_pressure")], marker="*", s=90, color=COLORS["orange"], edgecolors=COLORS["black"], linewidths=0.35, zorder=4)

    tick_positions = np.linspace(0, len(ordered) - 1, 7, dtype=int)
    ax.set_xticks(tick_positions, ordered.loc[tick_positions, "case_index"].astype(int))
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("已验证算例编号")
    ax.set_title("变量与响应随优化过程的演化", pad=6)
    ax.tick_params(top=False, right=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("归一化取值")
    save_figure(fig, "cylinder_parameter_evolution_heatmap")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_parameter_evolution_heatmap(design, best)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_parameter_evolution_heatmap.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
