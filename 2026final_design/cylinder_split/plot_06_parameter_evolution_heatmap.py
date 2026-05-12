"""
plot_06_parameter_evolution_heatmap.py

输出 PNG：figures/cylinder/cylinder_parameter_evolution_heatmap.png
功能：演化热图：观察参数与响应随优化迭代过程的变化（展示全部算例）。
"""

from cylinder_plot_common import *


# ============================== 对应 PNG：cylinder_parameter_evolution_heatmap.png ==============================
def draw_parameter_evolution_heatmap(design: pd.DataFrame, best: pd.Series) -> None:
    """演化热图：展示全部算例的演化过程。"""

    # 不再过滤 case_index <= 58，直接展示全部算例
    ordered = design.sort_values("case_index").reset_index(drop=True)

    if ordered.empty:
        raise ValueError("design 数据为空，无法绘制参数演化热图。")

    variables = ["t", "n", "p", "a", "total_mass", "buckling_pressure"]
    labels = ["t", "n", "p", "a", "质量", "屈曲特征值"]

    # 重新计算归一化矩阵
    matrix = ordered[variables].apply(normalize_to_unit).T.to_numpy(dtype=float)

    # 根据算例数量自动调整图宽，避免接近 100 个算例时太挤
    fig_width = max(9.5, min(16.0, 0.10 * len(ordered)))
    fig = plt.figure(figsize=(fig_width, 3.25), constrained_layout=True)

    grid = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.0])
    ax_strip = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_strip)

    # --- 上方阶段条绘制 ---
    stage_strip = np.where(
        ordered["stage"] == "LHS screening",
        0.35,
        0.80,
    )[None, :]

    ax_strip.imshow(stage_strip, aspect="auto", cmap="Greys", vmin=0, vmax=1)

    feasible_x = np.where(ordered["feasible"].to_numpy())[0]
    infeasible_x = np.where(~ordered["feasible"].to_numpy())[0]

    ax_strip.scatter(
        infeasible_x,
        np.zeros_like(infeasible_x),
        s=8,
        color=COLORS["gray"],
        alpha=0.45,
    )

    ax_strip.scatter(
        feasible_x,
        np.zeros_like(feasible_x),
        s=10,
        color=COLORS["green"],
        alpha=0.85,
    )

    # 检查最优解是否在当前 ordered 数据中
    if best["case_name"] in ordered["case_name"].values:
        best_pos = int(
            np.where(ordered["case_name"].to_numpy() == best["case_name"])[0][0]
        )

        ax_strip.scatter(
            [best_pos],
            [0],
            marker="*",
            s=75,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.35,
            zorder=4,
        )

        ax.scatter(
            [best_pos],
            [variables.index("total_mass")],
            marker="*",
            s=90,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.35,
            zorder=4,
        )

        ax.scatter(
            [best_pos],
            [variables.index("buckling_pressure")],
            marker="*",
            s=90,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.35,
            zorder=4,
        )

    # 阶段分界线：基于 INITIAL_LHS_LAST_CASE 自动定位
    lhs_indices = np.where(ordered["case_index"] <= INITIAL_LHS_LAST_CASE)[0]

    if lhs_indices.size > 0:
        split_pos = int(lhs_indices.max()) + 0.5
    else:
        split_pos = -0.5

    ax_strip.axvline(
        split_pos,
        color=COLORS["black"],
        lw=0.8,
        ls=(0, (3, 2)),
    )

    # 阶段文字位置动态计算，避免 case_index 不连续时标注错位
    n_case = len(ordered)

    lhs_center = 0.5 * max(split_pos, 0) / max(n_case - 1, 1)
    gpr_center = 0.5 * (split_pos + n_case - 1) / max(n_case - 1, 1)

    lhs_center = min(max(lhs_center, 0.08), 0.92)
    gpr_center = min(max(gpr_center, 0.08), 0.92)

    ax_strip.text(
        lhs_center,
        0.75,
        "LHS",
        transform=ax_strip.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["black"],
    )

    ax_strip.text(
        gpr_center,
        0.75,
        "GPR",
        transform=ax_strip.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color="white",
    )

    ax_strip.set_yticks([])
    ax_strip.set_ylabel("阶段", rotation=0, ha="right", va="center", fontsize=6.4)
    ax_strip.tick_params(axis="x", labelbottom=False, bottom=False)

    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    # --- 下方热图绘制 ---
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.axvline(
        split_pos,
        color="white",
        lw=1.0,
        ls=(0, (3, 2)),
    )

    # X 轴刻度设置：只控制标签数量，不控制实际显示的算例数量
    n_ticks = min(8, len(ordered))
    tick_indices = np.unique(
        np.linspace(0, len(ordered) - 1, n_ticks, dtype=int)
    )

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(ordered.loc[tick_indices, "case_index"].astype(int))

    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("算例编号")

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("归一化取值")

    beautify_axis(ax)
    save_figure(fig, "圆筒参数演化热图")
    plt.close(fig)


# ============================== 补全 main 函数 ==============================
def main() -> None:
    """单独运行本文件时，生成对应的 PNG。"""

    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()

    # 从 common 加载数据
    _, _, design, best, _ = load_data()

    draw_parameter_evolution_heatmap(design, best)

    cleanup_temp_files()

    print(f"Saved: {OUT_DIR / 'cylinder_parameter_evolution_heatmap.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()