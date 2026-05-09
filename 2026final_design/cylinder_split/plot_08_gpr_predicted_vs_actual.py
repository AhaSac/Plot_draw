"""
plot_08_gpr_predicted_vs_actual.py

输出 PNG：figures/cylinder/cylinder_gpr_predicted_vs_actual.png
功能：GPR 预测值-真实值验证图：展示五折交叉验证误差和置信区间。

修改提示：
- 本脚本只负责生成上面这 1 张 PNG；
- 图形样式、颜色、字体、数据读取等公共设置在 cylinder_plot_common.py 中；
- 想改坐标轴、标题、标注、颜色映射时，优先修改下方 draw_gpr_validation() 函数。
"""

from cylinder_plot_common import *



# ============================== 对应 PNG：cylinder_gpr_predicted_vs_actual.png ==============================
# 功能：GPR 预测值-真实值验证图：展示五折交叉验证误差和置信区间。
def draw_gpr_validation(validation: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """预测值 vs 真实值：带置信区间并标注 R2/RMSE/MAE。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)

    for ax, target in zip(axes, RESPONSE_COLS):
        subset = validation[validation["target"] == target].copy()
        target_metrics = metrics[metrics["target"] == target].iloc[0]
        x = subset["actual"].to_numpy(dtype=float)
        y = subset["predicted"].to_numpy(dtype=float)
        yerr = 1.96 * subset["std"].to_numpy(dtype=float)

        feasible = subset["feasible"].to_numpy(dtype=bool)
        ax.errorbar(
            x[~feasible],
            y[~feasible],
            yerr=yerr[~feasible],
            fmt="o",
            ms=3.0,
            mfc="white",
            mec=COLORS["gray"],
            ecolor=COLORS["light_gray"],
            elinewidth=0.55,
            capsize=1.6,
            alpha=0.82,
            label="不可行",
        )
        ax.errorbar(
            x[feasible],
            y[feasible],
            yerr=yerr[feasible],
            fmt="o",
            ms=3.2,
            mfc=COLORS["green"],
            mec="white",
            ecolor=COLORS["green"],
            elinewidth=0.55,
            capsize=1.6,
            alpha=0.88,
            label="可行",
        )

        low = min(np.nanmin(x), np.nanmin(y))
        high = max(np.nanmax(x), np.nanmax(y))
        pad = (high - low) * 0.08
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=COLORS["black"], lw=0.85, ls=(0, (3, 2)))
        ax.fill_between(
            [low - pad, high + pad],
            [low - pad - 0.1 * (high - low), high + pad - 0.1 * (high - low)],
            [low - pad + 0.1 * (high - low), high + pad + 0.1 * (high - low)],
            color=COLORS["light_gray"],
            alpha=0.25,
            linewidth=0,
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"真实{RESPONSE_LABELS[target]}")
        ax.set_ylabel(f"预测{RESPONSE_LABELS[target]}")
        ax.set_title(RESPONSE_LABELS[target])
        ax.text(
            0.05,
            0.95,
            (
                f"五折交叉验证\n"
                f"$R^2$ = {target_metrics['r2']:.3f}\n"
                f"RMSE = {target_metrics['rmse']:.3g}\n"
                f"MAE = {target_metrics['mae']:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COLORS["light_gray"], "lw": 0.6},
        )
        beautify_axis(ax)

    axes[0].legend(loc="lower right")
    fig.suptitle("高斯过程回归模型验证：预测值与真实值", y=1.03, fontsize=8.4, fontweight="bold")
    save_figure(fig, "cylinder_gpr_predicted_vs_actual")
    plt.close(fig)


def main() -> None:
    """单独运行本文件时，只生成对应的 1 张 PNG。"""
    cleanup_non_png_outputs()
    style_name = try_enable_nature_style()
    _, baseline, design, best, pressure_limit = load_data()
    validation, metrics = cross_validated_gpr_predictions(design, RESPONSE_COLS)
    draw_gpr_validation(validation, metrics)
    metrics_path = OUT_DIR / "cylinder_gpr_validation_metrics.csv"
    validation_path = OUT_DIR / "cylinder_gpr_validation_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    validation.to_csv(validation_path, index=False)
    cleanup_temp_files()
    print(f"Saved: {OUT_DIR / 'cylinder_gpr_predicted_vs_actual.png'}")
    print(f"Saved CSV: {metrics_path}")
    print(f"Saved CSV: {validation_path}")
    print(f"Style: {style_name}")


if __name__ == "__main__":
    main()
