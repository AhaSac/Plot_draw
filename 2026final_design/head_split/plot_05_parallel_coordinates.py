"""
plot_05_parallel_coordinates.py

输出 PNG：figures/head/参数平行优化变化.png
功能：平行坐标图：展示高维参数组合、响应指标和最优解轨迹。
"""

from head_plot_common import *


# ============================== 对应 PNG：参数平行优化变化.png ==============================
# 功能：平行坐标图：展示高维参数组合、响应指标和最优解轨迹。
def draw_parallel_coordinates(design: pd.DataFrame, best: pd.Series) -> None:
    """归一化平行坐标图。"""
    columns = PARAMETER_COLS + RESPONSE_COLS
    normalized = pd.DataFrame({col: normalize_to_unit(design[col]) for col in columns})
    best_norm = {col: float(normalize_to_unit(design[col]).loc[best.name]) for col in columns}
    x = np.arange(len(columns))

    fig, ax = plt.subplots(figsize=(9.0, 3.5), constrained_layout=True)

    for _, row in normalized.iterrows():
        color = COLORS["green"] if bool(design.loc[row.name, "feasible"]) else COLORS["gray"]
        alpha = 0.16 if bool(design.loc[row.name, "feasible"]) else 0.07
        ax.plot(x, row.to_numpy(dtype=float), color=color, alpha=alpha, lw=0.75)

    ax.plot(x, [best_norm[col] for col in columns], color=COLORS["orange"], lw=2.0, marker="o", ms=3.0, zorder=5)

    ax.set_xticks(x, [PARAMETER_LABELS.get(col, RESPONSE_LABELS.get(col, col)) for col in columns], rotation=45, ha="right")
    ax.set_ylabel("归一化值")
    ax.set_ylim(0, 1.02)
    ax.set_title("平行坐标对比")
    beautify_axis(ax, minor=False)

    legend_handles = [
        Line2D([0], [0], color=COLORS["green"], lw=1.2, alpha=0.7, label="可行方案"),
        Line2D([0], [0], color=COLORS["gray"], lw=1.2, alpha=0.4, label="不可行方案"),
        Line2D([0], [0], color=COLORS["orange"], lw=2.0, label="最优解"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    save_figure(fig, "参数平行优化变化")
    plt.close(fig)


def main() -> None:
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_parallel_coordinates(design, best)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '参数平行优化变化.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
