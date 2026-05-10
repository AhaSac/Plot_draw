"""
plot_09_gpr_response_surface.py

输出 PNG：figures/head/高斯过程响应面.png
功能：GPR 二维响应面图：固定其余参数为最优解，展示质量和屈曲压力的代理模型等高线。
"""

from head_plot_common import *


# ============================== 对应 PNG：高斯过程响应面.png ==============================
# 功能：GPR 二维响应面图：固定其余参数为最优解，展示质量和屈曲压力的代理模型等高线。
def draw_gpr_response_surface(
    design: pd.DataFrame,
    best: pd.Series,
    models: dict[str, SimpleGaussianProcess],
    sensitivity: pd.DataFrame,
    pressure_limit: float,
) -> None:
    """二维响应面/等高线图：固定其余参数为最优解取值。"""
    x_col, y_col = response_surface_parameters(sensitivity)
    grid_n = 120

    x_min, x_max = design[x_col].min(), design[x_col].max()
    y_min, y_max = design[y_col].min(), design[y_col].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    x_values = np.linspace(x_min - x_pad, x_max + x_pad, grid_n)
    y_values = np.linspace(y_min - y_pad, y_max + y_pad, grid_n)
    xx, yy = np.meshgrid(x_values, y_values)

    surface_frame = pd.DataFrame({col: np.full(xx.size, best[col]) for col in PARAMETER_COLS})
    surface_frame[x_col] = xx.ravel()
    surface_frame[y_col] = yy.ravel()
    surface_frame = enforce_integer_parameters(surface_frame)
    valid_mask = physical_feasible(surface_frame)

    predictions = {}
    for target, model in models.items():
        pred, _ = model.predict(surface_frame[PARAMETER_COLS].to_numpy(dtype=float))
        pred[~valid_mask] = np.nan
        predictions[target] = pred.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95), constrained_layout=True)
    cmap_by_target = {"total_mass": "viridis_r", "buckling_pressure": "viridis"}

    for ax, target in zip(axes, RESPONSE_COLS):
        zz = predictions[target]
        contour = ax.contourf(xx, yy, zz, levels=16, cmap=cmap_by_target[target])
        ax.contour(xx, yy, zz, levels=8, colors="white", linewidths=0.35, alpha=0.75)
        if target == "buckling_pressure":
            ax.contour(xx, yy, zz, levels=[pressure_limit], colors=COLORS["black"], linewidths=1.0, linestyles="--")

        ax.scatter(
            design[x_col],
            design[y_col],
            s=12,
            facecolors="none",
            edgecolors=np.where(design["feasible"], COLORS["green"], COLORS["gray"]),
            linewidths=0.45,
            alpha=0.72,
        )
        ax.scatter(
            best[x_col],
            best[y_col],
            marker="*",
            s=130,
            color=COLORS["orange"],
            edgecolors=COLORS["black"],
            linewidths=0.45,
            zorder=5,
        )
        ax.set_xlabel(PARAMETER_LABELS[x_col])
        ax.set_ylabel(PARAMETER_LABELS[y_col])
        ax.set_title(f"高斯过程响应面：{RESPONSE_LABELS[target]}")
        if x_col in INTEGER_PARAMETER_COLS:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if y_col in INTEGER_PARAMETER_COLS:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(RESPONSE_LABELS[target])
        beautify_axis(ax)

   # fixed = ", ".join([f"{col}={best[col]:.3g}" for col in PARAMETER_COLS if col not in {x_col, y_col}])
  #  fig.suptitle(f"固定其余参数：{fixed}", fontsize=7.0, y=1.03)

    save_figure(fig, "高斯过程响应面")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    gpr_models = fit_gpr_models(design, RESPONSE_COLS)
    sensitivity = gp_permutation_sensitivity(design, gpr_models)
    draw_gpr_response_surface(design, best, gpr_models, sensitivity, pressure_limit)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / '高斯过程响应面.png'}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
