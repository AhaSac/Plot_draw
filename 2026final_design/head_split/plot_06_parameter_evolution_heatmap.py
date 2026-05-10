"""
plot_06_parameter_evolution_heatmap.py

输出 PNG：figures/head/封头参数演化热图.png
功能：演化热图：观察参数与响应随优化迭代过程的变化。
"""

from head_plot_common import *


# ============================== 对应 PNG：封头参数演化热图.png ==============================
# 功能：演化热图：观察参数与响应随优化迭代过程的变化。
def draw_parameter_evolution_heatmap(design: pd.DataFrame, best: pd.Series) -> None:
    """按算例顺序展示参数和响应的归一化热图。"""
    ordered = design.sort_values("case_index").reset_index(drop=True)
    heat_cols = PARAMETER_COLS + RESPONSE_COLS
    heat = np.column_stack([normalize_to_unit(ordered[col]).to_numpy(dtype=float) for col in heat_cols])

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    image = ax.imshow(heat, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_xticks(np.arange(len(heat_cols)), [PARAMETER_LABELS.get(col, RESPONSE_LABELS.get(col, col)) for col in heat_cols], rotation=45, ha="right")
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("归一化值")



    save_figure(fig, "封头参数演化热图")
    plt.close(fig)


def main() -> None:
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    draw_parameter_evolution_heatmap(design, best)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '封头参数演化热图.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
